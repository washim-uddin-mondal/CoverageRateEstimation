import time
import numpy as np
from Parameters import ParseInput
import matplotlib.pyplot as plt
import DataProcessing
import ForwardProblem
import InverseProblem
from FolderCreation import CreateBasicFolders
import logging

if __name__ == '__main__':

    args = ParseInput()
    t0 = time.time()

    # Indexed in the same order in forward-problem
    args.RelativePlots = ['PPP', 'Best']
    args.AbsolutePlots = ['CNN', 'PPP', 'Best']

    CreateBasicFolders(args)

    # Logging
    args.logFileName = args.RawResults + 'progress.log'
    open(args.logFileName, 'w').close()
    logging.basicConfig(filename=args.logFileName,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    args.logger = logging.getLogger()
    args.logger.setLevel(logging.INFO)

    args.logger.info('Data processing is in progress.')
    DataProcessing.data_process(args)

    SINRThrDiv = (args.MaxSINRThr-args.MinSINRThr)/args.NumSINRThr
    args.SINRThrVec = args.MinSINRThr + SINRThrDiv * np.array(range(args.NumSINRThr))

    NumBox = np.load(args.ModelsFolder + 'NumBox.npy')

    PercentErrorReduction = np.zeros([len(args.RelativePlots), args.NumSINRThr])
    SqPercentErrorReduction = np.zeros([len(args.RelativePlots), args.NumSINRThr])
    Err = np.zeros([len(args.AbsolutePlots), args.NumSINRThr])
    SqErr = np.zeros([len(args.AbsolutePlots), args.NumSINRThr])

    np.save(args.RawResults + 'PercentErrorReduction.npy', PercentErrorReduction)
    np.save(args.RawResults + 'SqPercentErrorReduction.npy', SqPercentErrorReduction)
    np.save(args.RawResults + 'AvgError.npy', Err)
    np.save(args.RawResults + 'SqError.npy', SqErr)

    # training curve indices
    IndexSet = np.array(range(int(args.NumSINRThr/2)))*2

    plt.figure()
    plt.ylabel("Training Loss")
    plt.xlabel("Number of mini batches")

    MeanTrainLossVec = np.zeros([args.NumSINRThr, args.TrainReRun*int(int(args.TrainFraction*NumBox)/args.batchlength)])
    MeanSqTrainLossVec = np.zeros_like(MeanTrainLossVec)
    np.save(args.RawResults + f'MeanTrainLossVec.npy', MeanTrainLossVec)
    np.save(args.RawResults + f'MeanSqTrainLossVec.npy', MeanSqTrainLossVec)

    for SeedIndex in range(args.MaxSeedNum):
        args.CurrentSeedIndex = SeedIndex

        args.logger.info('Training is in progress.')
        ForwardProblem.train(args)

        args.logger.info('Evaluation is in progress.')
        ForwardProblem.evaluate(args)

    M = np.load(args.RawResults + f'MeanTrainLossVec.npy') / args.MaxSeedNum
    SM = np.load(args.RawResults + f'MeanSqTrainLossVec.npy') / args.MaxSeedNum
    sd = (np.maximum(SM - M ** 2, 0)) ** 0.5  # Negative values may appear due to precision error

    for SINRThrIndex in IndexSet:
        SINRThr = args.SINRThrVec[SINRThrIndex]
        plt.plot(M[SINRThrIndex, :], label=f'Threshold: {SINRThr} dB')
        plt.fill_between(range(len(M[SINRThrIndex, :])), M[SINRThrIndex, :] - sd[SINRThrIndex, :], M[SINRThrIndex, :] + sd[SINRThrIndex, :], alpha=0.3)

    plt.legend()
    plt.savefig(args.ResultsFolder + 'TrainingLoss.png')

    plt.figure()
    plt.ylabel('Percentage of Loss Reduction')
    plt.xlabel('SINR Threshold (dB)')

    ErrRed = np.load(args.RawResults + 'PercentErrorReduction.npy')
    SqErrRed = np.load(args.RawResults + 'SqPercentErrorReduction.npy')

    for RelativePlotIndex in range(len(args.RelativePlots)):
        AvgErr = ErrRed[RelativePlotIndex]
        sdErr = (np.maximum(SqErrRed[RelativePlotIndex] - AvgErr ** 2, 0)) ** 0.5
        plt.plot(args.SINRThrVec, AvgErr, label=args.RelativePlots[RelativePlotIndex])
        plt.fill_between(args.SINRThrVec, AvgErr - sdErr, AvgErr + sdErr, alpha=0.3)
    plt.legend()
    plt.savefig(args.ResultsFolder + 'EvaluatedLossReduction.png')

    plt.figure()
    plt.ylabel('Evaluated Loss')
    plt.xlabel('SINR Threshold (dB)')
    Err = np.load(args.RawResults + 'AvgError.npy')
    SqErr = np.load(args.RawResults + 'SqError.npy')

    for AbsolutePlotIndex in range(len(args.AbsolutePlots)):
        AvgErr = Err[AbsolutePlotIndex]
        sdErr = (np.maximum(SqErr[AbsolutePlotIndex] - AvgErr**2, 0))**0.5
        plt.plot(args.SINRThrVec, AvgErr, label=args.AbsolutePlots[AbsolutePlotIndex])
        plt.fill_between(args.SINRThrVec, AvgErr - sdErr, AvgErr + sdErr, alpha=0.3)
    plt.legend()
    plt.savefig(args.ResultsFolder + 'EvaluatedLoss.png')

    # Inverse Problem
    if args.inverse:
        InverseProblem.AM(args)

    t1 = time.time()
    args.logger.info(f'Elapsed time is: {t1 - t0} sec.')
