import torch
import torch.optim as optim
import numpy as np
import copy
from CoverageRateSimulator import CoverageRateSimulator
from NeuralNets import Encoder, Decoder
from UsefulFunctions import PPPCoverageFunction, PPPRateFunction


def train(args):
    xLength = args.xLength
    xSteps = args.xSteps
    yLength = args.yLength
    ySteps = args.ySteps

    # Load Data
    rel_x = torch.tensor(np.load(args.ModelsFolder + 'rel_x.npy'))
    rel_y = torch.tensor(np.load(args.ModelsFolder + 'rel_y.npy'))
    BoxIndex = np.load(args.ModelsFolder + 'BoxIndex.npy')
    BoxCardinality = np.load(args.ModelsFolder + 'BoxCardinality.npy')
    EndBSIndex = np.load(args.ModelsFolder + 'EndBSIndex.npy')

    NumBox = np.size(BoxIndex)
    BoxIndex = np.random.permutation(BoxIndex)
    np.save(args.ModelsFolder + f'BoxIndex.npy', BoxIndex)
    TrainBoxIndex = BoxIndex[:int(args.TrainFraction*NumBox)]

    Simulator = CoverageRateSimulator(args)

    EncNet = []             # list of Encoders for all SINRThr values
    DecNet = []             # list of Decoders for all SINRThr values
    optimizers = []

    for SINRThrIndex in range(args.NumSINRThr):
        EncNet.append(Encoder(args))
        DecNet.append(Decoder(args))
        params_to_optimize = list(EncNet[-1].parameters()) + list(DecNet[-1].parameters())
        optimizers.append(optim.Adam(params=params_to_optimize, lr=args.lr))

    # Either coverage or rate matrix depending on user input
    SimOutMatrix = torch.zeros(args.batchlength, args.NumSINRThr, int(xSteps*args.MaskRatio), int(ySteps*args.MaskRatio))

    avgL = torch.zeros(args.NumSINRThr)              # Average Cumulative Loss for each SINRThr values
    iteration = 0

    LossVec = torch.zeros([args.NumSINRThr, args.TrainReRun*int(int(args.TrainFraction*NumBox)/args.batchlength)])

    for _ in range(args.TrainReRun):
        data_index = 0
        TrainBoxIndex = np.random.permutation(TrainBoxIndex)
        while data_index < len(TrainBoxIndex) - args.batchlength:
            iteration += 1

            # BSCells must be of 4 dimension for it to act as an input to the CNN
            BSCells = torch.zeros(args.batchlength, 1, xSteps, ySteps)

            for batchNo in range(args.batchlength):
                box_index = TrainBoxIndex[data_index]
                NumberBS = BoxCardinality[box_index]

                xBS = rel_x[EndBSIndex[box_index] - NumberBS: EndBSIndex[box_index]]
                xCells = (xBS/(xLength/xSteps)).long()
                yBS = rel_y[EndBSIndex[box_index] - NumberBS: EndBSIndex[box_index]]
                yCells = (yBS / (yLength / ySteps)).long()

                BSCells[batchNo, 0, xCells, yCells] = 1

                SimOutMatrix[batchNo, :, :, :] = Simulator.simulate(NumberBS, xBS, yBS)
                data_index += 1

            for SINRThrIndex in range(args.NumSINRThr):
                EncOutput = EncNet[SINRThrIndex](BSCells)
                DecOutput = DecNet[SINRThrIndex](EncOutput)

                MaskedDecOutput = copy.copy(DecOutput[:, :, args.xIndexMin:args.xIndexMax, args.yIndexMin:args.yIndexMax])
                SimOutput = SimOutMatrix[:, SINRThrIndex, :, :].unsqueeze(1)
                batchLoss = torch.mean(torch.abs(MaskedDecOutput - SimOutput) ** args.loss_exp)

                optimizers[SINRThrIndex].zero_grad()
                batchLoss.backward()
                optimizers[SINRThrIndex].step()

                avgL[SINRThrIndex] += (batchLoss - avgL[SINRThrIndex])/iteration
                LossVec[SINRThrIndex, iteration-1] = avgL[SINRThrIndex]**(1/args.loss_exp)

            if iteration % 100 == 0:
                args.logger.info(f'Data: {args.Country}, Seed Index: {args.CurrentSeedIndex}, Iteration: {iteration}')

    for SINRThrIndex in range(args.NumSINRThr):
        torch.save(EncNet[SINRThrIndex].state_dict(), args.ModelsFolder + f'Encoder{SINRThrIndex}.pkl')
        torch.save(DecNet[SINRThrIndex].state_dict(), args.ModelsFolder + f'Decoder{SINRThrIndex}.pkl')

    MeanTrainLossVec = np.load(args.RawResults+f'MeanTrainLossVec.npy')
    MeanSqTrainLossVec = np.load(args.RawResults + f'MeanSqTrainLossVec.npy')
    MeanTrainLossVec += np.array(LossVec.detach())
    MeanSqTrainLossVec += np.array(LossVec.detach())**2
    np.save(args.RawResults + f'MeanTrainLossVec.npy', MeanTrainLossVec)
    np.save(args.RawResults + f'MeanSqTrainLossVec.npy', MeanSqTrainLossVec)


def evaluate(args):
    xLength = args.xLength
    xSteps = args.xSteps
    yLength = args.yLength
    ySteps = args.ySteps

    # Load Data
    rel_x = torch.tensor(np.load(args.ModelsFolder + 'rel_x.npy'))
    rel_y = torch.tensor(np.load(args.ModelsFolder + 'rel_y.npy'))
    BoxIndex = np.load(args.ModelsFolder + f'BoxIndex.npy')
    BoxCardinality = np.load(args.ModelsFolder + 'BoxCardinality.npy')
    EndBSIndex = np.load(args.ModelsFolder + 'EndBSIndex.npy')

    NumBox = np.size(BoxIndex)
    TestBoxIndex = BoxIndex[int(args.TrainFraction * NumBox):]

    Simulator = CoverageRateSimulator(args)

    EncNet = []
    DecNet = []
    for SINRThrIndex in range(args.NumSINRThr):
        EncNet.append(Encoder(args))
        EncNet[-1].load_state_dict(torch.load(args.ModelsFolder + f'Encoder{SINRThrIndex}.pkl'))
        DecNet.append(Decoder(args))
        DecNet[-1].load_state_dict(torch.load(args.ModelsFolder + f'Decoder{SINRThrIndex}.pkl'))

    avgL = np.zeros([len(args.AbsolutePlots), args.NumSINRThr])

    data_index = 0

    while data_index < len(TestBoxIndex):
        box_index = TestBoxIndex[data_index]
        NumberBS = BoxCardinality[box_index]

        DensityBS = NumberBS/(xLength*yLength)

        xBS = rel_x[EndBSIndex[box_index] - NumberBS: EndBSIndex[box_index]]
        xCells = (xBS / (xLength / xSteps)).long()
        yBS = rel_y[EndBSIndex[box_index] - NumberBS: EndBSIndex[box_index]]
        yCells = (yBS / (yLength / ySteps)).long()

        BSCells = torch.zeros(1, 1, xSteps, ySteps)
        BSCells[0, 0, xCells, yCells] = 1

        SimOutMatrix = Simulator.simulate(NumberBS, xBS, yBS)

        lossCNN = torch.zeros(args.NumSINRThr)
        lossPPP = torch.zeros(args.NumSINRThr)
        lossBest = torch.zeros(args.NumSINRThr)

        for SINRThrIndex in range(args.NumSINRThr):
            EncOutput = EncNet[SINRThrIndex](BSCells)
            DecOutput = DecNet[SINRThrIndex](EncOutput)
            MaskedDecOutput = DecOutput[0, 0, args.xIndexMin:args.xIndexMax, args.yIndexMin:args.yIndexMax]

            SimOutput = SimOutMatrix[SINRThrIndex, :, :]

            if args.coverage:
                SINRThr = args.SINRThrVec[SINRThrIndex]
                PPPAvg = PPPCoverageFunction(DensityBS, SINRThr, args)
            else:
                PPPAvg = PPPRateFunction(DensityBS, args)

            lossCNN[SINRThrIndex] = torch.mean(torch.abs(MaskedDecOutput - SimOutput) ** args.loss_exp)
            lossPPP[SINRThrIndex] = torch.mean(torch.abs(SimOutput - PPPAvg) ** args.loss_exp)
            lossBest[SINRThrIndex] = torch.mean(torch.abs(SimOutput - torch.mean(SimOutput)) ** args.loss_exp)

        data_index += 1
        avgL[0, :] += (np.array(lossCNN.detach()) - avgL[0, :]) / data_index
        avgL[1, :] += (np.array(lossPPP.detach()) - avgL[1, :]) / data_index
        avgL[2, :] += (np.array(lossBest.detach()) - avgL[2, :]) / data_index

    args.logger.info(f'Data: {args.Country}, Seed Index: {args.CurrentSeedIndex}')

    Err = np.load(args.RawResults + 'AvgError.npy')
    SqErr = np.load(args.RawResults + 'SqError.npy')
    ErrRed = np.load(args.RawResults + 'PercentErrorReduction.npy')
    SqErrRed = np.load(args.RawResults + 'SqPercentErrorReduction.npy')

    for RelativePlotIndex in range(len(args.RelativePlots)):
        PerRed = 100 * (1 - (avgL[0, :] ** (1 / args.loss_exp) / avgL[RelativePlotIndex+1, :] ** (1 / args.loss_exp)))
        ErrRed[RelativePlotIndex, :] += (PerRed - ErrRed[RelativePlotIndex, :]) / (args.CurrentSeedIndex + 1)
        SqErrRed[RelativePlotIndex, :] += (PerRed ** 2 - SqErrRed[RelativePlotIndex, :]) / (args.CurrentSeedIndex + 1)

    np.save(args.RawResults + 'PercentErrorReduction.npy', ErrRed)
    np.save(args.RawResults + 'SqPercentErrorReduction.npy', SqErrRed)

    for AbsolutePlotIndex in range(len(args.AbsolutePlots)):
        Err[AbsolutePlotIndex, :] += (np.array(avgL[AbsolutePlotIndex, :] ** (1 / args.loss_exp)) - Err[AbsolutePlotIndex, :]) / (args.CurrentSeedIndex + 1)
        SqErr[AbsolutePlotIndex, :] += (np.array(avgL[AbsolutePlotIndex, :] ** (2 / args.loss_exp)) - SqErr[AbsolutePlotIndex, :]) / (args.CurrentSeedIndex + 1)

    np.save(args.RawResults + 'AvgError.npy', Err)
    np.save(args.RawResults + 'SqError.npy', SqErr)
