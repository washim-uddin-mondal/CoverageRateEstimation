import os


def CreateBasicFolders(args):

    if not os.path.exists(args.ModelsFolder):
        os.makedirs(args.ModelsFolder)

    if not os.path.exists(args.RawResults):
        os.makedirs(args.RawResults)

    if not os.path.exists(args.InverseResultsFolder):
        os.makedirs(args.InverseResultsFolder)
