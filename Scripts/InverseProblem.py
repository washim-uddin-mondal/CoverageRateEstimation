import torch
import numpy as np
from NeuralNets import Encoder, Decoder
import matplotlib.pyplot as plt


def AM(args):                                 # Alternative Maximization
    ModelsFolder = args.Models + args.Country + '/'
    SINRThrIndex = int((args.SINRThrInv - args.MinSINRThr)/((args.MaxSINRThr - args.MinSINRThr)/args.NumSINRThr))

    if (SINRThrIndex < 0) or (SINRThrIndex >= args.NumSINRThr):
        raise ValueError('Chosen SINR threshold for the inverse problem is outside of training range.')

    # Load Data
    rel_x = torch.tensor(np.load(ModelsFolder + 'rel_x.npy'))
    rel_y = torch.tensor(np.load(ModelsFolder + 'rel_y.npy'))
    BoxIndex = np.load(ModelsFolder + 'BoxIndex.npy')
    BoxCardinality = np.load(ModelsFolder + 'BoxCardinality.npy')
    EndBSIndex = np.load(ModelsFolder + 'EndBSIndex.npy')

    EncNet = Encoder(args)
    EncNet.load_state_dict(torch.load(ModelsFolder + f'Encoder{SINRThrIndex}.pkl'))
    DecNet = Decoder(args)
    DecNet.load_state_dict(torch.load(ModelsFolder + f'Decoder{SINRThrIndex}.pkl'))

    def ComputeObj(BSCell):
        x = EncNet(BSCell)
        x = DecNet(x)
        x = x[0, 0, args.xIndexMin:args.xIndexMax, args.yIndexMin:args.yIndexMax]
        return torch.mean(torch.tensor(x > args.CoverageLBInv)*1.0)

    def DrawHeatMap():
        x = EncNet(BSCells)
        x = DecNet(x)
        x = x[0, 0, args.xIndexMin:args.xIndexMax, args.yIndexMin:args.yIndexMax].detach().numpy()

        plt.figure(figsize=(7, 6))
        plt.pcolormesh(x)
        plt.colorbar()
        plt.xlim([-int(args.xSteps/2) + int(int(args.xSteps / 2)/args.MaskRatio), int(args.xSteps/2) + int(int(args.xSteps / 2)/args.MaskRatio)])
        plt.ylim([-int(args.ySteps/2) + int(int(args.ySteps / 2)/args.MaskRatio), int(args.ySteps/2) + int(int(args.ySteps / 2)/args.MaskRatio)])
        plt.xticks([])
        plt.yticks([])

        plt.scatter(xCellsOld - int(args.xSteps / 2) + int(int(args.xSteps / 2)/args.MaskRatio), yCellsOld - int(args.ySteps / 2) + int(int(args.ySteps / 2)/args.MaskRatio), marker='o', c='red')
        plt.scatter(xCellsNew - int(args.xSteps / 2) + int(int(args.xSteps / 2)/args.MaskRatio), yCellsNew - int(args.ySteps / 2) + int(int(args.ySteps / 2)/args.MaskRatio), marker='*', c='red')

        if NewNumBS == 0:
            plt.title('Before deployment')
            plt.savefig('Results/Inverse/' + args.Country + f'/Old{OldNumBS}New{NewNumBS}.png')

        if Val >= args.UserFracThrInv or NewNumBS == args.MaxDeployInv - 1:
            plt.title('After deployment')
            plt.savefig('Results/Inverse/' + args.Country + f'/Old{OldNumBS}New{NewNumBS}.png')

    xLength = args.xLength
    yLength = args.yLength
    xSteps = args.xSteps
    ySteps = args.ySteps

    BoxIndex = np.random.permutation(BoxIndex)
    ChosenBoxIndex = BoxIndex[0]                     # Randomly choose a box
    OldNumBS = BoxCardinality[ChosenBoxIndex]
    xBSOld = rel_x[EndBSIndex[ChosenBoxIndex] - OldNumBS: EndBSIndex[ChosenBoxIndex]]  # x coordinates of existing BSs
    xCellsOld = (xBSOld / (xLength / xSteps)).long()
    yBSOld = rel_y[EndBSIndex[ChosenBoxIndex] - OldNumBS: EndBSIndex[ChosenBoxIndex]]  # y coordinates of existing BSs
    yCellsOld = (yBSOld / (yLength / ySteps)).long()

    for NewNumBS in range(args.MaxDeployInv):
        for RandInitIndex in range(args.MaxRandInitInv):
            BSCells = torch.zeros(1, 1, xSteps, ySteps)
            xCellsNew = torch.randint(0, xSteps, [NewNumBS])
            yCellsNew = torch.randint(0, ySteps, [NewNumBS])
            BSCells[0, 0, xCellsNew, yCellsNew] = 1            # Random Initialization for new BSs
            BSCells[0, 0, xCellsOld, yCellsOld] = 1

            Val = ComputeObj(BSCells)
            Improvement = True

            while Improvement:
                Improvement = False

                for BSIndex in range(NewNumBS):

                    # Maximization along X-axis
                    BSCells[0, 0, xCellsNew[BSIndex], yCellsNew[BSIndex]] = 0
                    for xIndex in range(xSteps):
                        BSCells[0, 0, xIndex, yCellsNew[BSIndex]] = 1
                        tempVal = ComputeObj(BSCells)
                        BSCells[0, 0, xIndex, yCellsNew[BSIndex]] = 0

                        if tempVal > Val:
                            Improvement = True
                            xCellsNew[BSIndex] = xIndex
                            Val = tempVal

                    BSCells[0, 0, xCellsNew[BSIndex], yCellsNew[BSIndex]] = 1

                    # Maximization along Y-axis
                    BSCells[0, 0, xCellsNew[BSIndex], yCellsNew[BSIndex]] = 0
                    for yIndex in range(ySteps):
                        BSCells[0, 0, xCellsNew[BSIndex], yIndex] = 1
                        tempVal = ComputeObj(BSCells)
                        BSCells[0, 0, xCellsNew[BSIndex], yIndex] = 0

                        if tempVal > Val:
                            Improvement = True
                            yCellsNew[BSIndex] = yIndex
                            Val = tempVal

                    BSCells[0, 0, xCellsNew[BSIndex], yCellsNew[BSIndex]] = 1

        args.logger.info(f'Existing BSs: {OldNumBS},  New Deployed BSs: {NewNumBS}.')
        args.logger.info(f'Maximum fraction of cells with coverage above the threshold is: {Val}.')

        DrawHeatMap()

        if Val >= args.UserFracThrInv:
            break
