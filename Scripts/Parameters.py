"""Contains all parameters related to the network"""
import argparse
import torch


def checkPositiveInt(value):
    try:
        iValue = int(value)
    except:
        raise ValueError(f'The input {value} must be an integer')

    if not iValue > 0:
        raise ValueError(f'The input {value} must be positive')

    return iValue


def checkPositiveFloat(value):
    try:
        fValue = float(value)
    except:
        raise ValueError(f'The input {value} must be a real number')

    if not fValue > 0:
        raise ValueError(f'The input {value} must be positive')

    return fValue


def checkFraction(value):
    try:
        fValue = float(value)
    except:
        raise ValueError(f'The input {value} must be a real number')

    if fValue < 0 or fValue > 1:
        raise ValueError(f'The input {value} must lie between 0 and 1')

    return fValue


def ParseInput():

    parser = argparse.ArgumentParser(description='System Parameters')

    """
    ====================== Command Line Options for the Simulator ===========================
    """
    # Options for the Rayleigh faded channel with no consideration of shadowing, LOS/NLOS paths and antenna gain
    parser.add_argument('--pathloss', type=checkPositiveFloat, default=4, dest='alpha', help='pathloss coefficient')
    parser.add_argument('--fading_shape', type=checkPositiveFloat, default=1.0, dest='m_par', help='shape parameter for the Nakagami distribution')
    parser.add_argument('--inverse_snr', type=checkPositiveFloat, default=10**(-2), dest='NoiseOverPower', help='inverse snr')

    # Options for realistic channel models
    parser.add_argument('--real_channel_model', action='store_true', help='use realistic channel models')
    parser.add_argument('--pathloss_los', type=checkPositiveFloat, default=2, dest='alpha_los', help='pathloss coefficient for LOS path')
    parser.add_argument('--pathloss_nlos', type=checkPositiveFloat, default=4, dest='alpha_nlos', help='pathloss coefficient for NLOS path')
    parser.add_argument('--fading_shape_los', type=checkPositiveFloat, default=3.0, dest='m_par_los', help='shape parameter for Nakagami distributed LOS path')
    parser.add_argument('--fading_shape_nlos', type=checkPositiveFloat, default=2.0, dest='m_par_nlos', help='shape parameter for Nakagami distributed NLOS path')
    parser.add_argument('--shadowing_sd', type=checkPositiveFloat, default=10, dest='sigma_shadow', help='standard deviation (linear) of log-normal shadowing')
    parser.add_argument('--beta_inv', type=checkPositiveFloat, default=140, dest='beta_inv', help='average LOS range')
    parser.add_argument('--main_lobe_t', type=checkPositiveFloat, default=10, dest='main_lobe_t', help='main lobe gain (linear) of transmitter')
    parser.add_argument('--side_lobe_t', type=checkPositiveFloat, default=0.1, dest='side_lobe_t', help='side lobe gain (linear) of transmitter')
    parser.add_argument('--beamwidth_t', type=checkPositiveFloat, default=45, dest='beamwidth_t', help='beamwidth (degree) of transmitter')
    parser.add_argument('--main_lobe_r', type=checkPositiveFloat, default=10, dest='main_lobe_r', help='main lobe gain (linear) of receiver')
    parser.add_argument('--side_lobe_r', type=checkPositiveFloat, default=0.1, dest='side_lobe_r', help='side lobe gain (linear) of receiver')
    parser.add_argument('--beamwidth_r', type=checkPositiveFloat, default=90, dest='beamwidth_r', help='beamwidth (degree) of receiver')

    parser.add_argument('--rate_norm_factor', type=checkPositiveFloat, default=10, dest='RateNormalizingFactor', help='rate normalizing factor')
    parser.add_argument('--max_iterations', type=checkPositiveInt, default=30, dest='MaxSimIter', help='maximum iterations to simulate the environment')

    # Options for an RoI
    parser.add_argument('--lengthX', type=checkPositiveFloat, default=10, dest='xLength', help='xlength (in km) of an RoI')
    parser.add_argument('--lengthY', type=checkPositiveFloat, default=10, dest='yLength', help='ylength (in km) of an RoI')
    parser.add_argument('--stepsX', type=checkPositiveInt, default=64, dest='xSteps', help='discretisation levels in x-direction of an RoI')
    parser.add_argument('--stepsY', type=checkPositiveInt, default=64, dest='ySteps', help='discretisation levels in y-direction of an RoI')
    parser.add_argument('--mask', type=checkFraction, default=0.5, dest='MaskRatio', help='ratio of RoE to RoI lengths')

    """ 
    =============================== Command Line Options for Data ===================================
    """
    parser.add_argument('--visualise', action='store_true', help='enable visualisation of data')
    parser.add_argument('--country', type=str, default='India', dest='Country', help='the country being considered')
    parser.add_argument('--min_bs', type=checkPositiveInt, default=20, dest='MinBS', help='minimum number of BSs in an RoI')
    parser.add_argument('--max_bs', type=checkPositiveInt, default=400, dest='MaxBS', help='maximum number of BSs in an RoI')
    parser.add_argument('--train_fraction', type=checkFraction, default=0.8, dest='TrainFraction', help='fraction of data used for training')

    """
    ============================ Command Line Options for Neural Networks ===============================
    """
    parser.add_argument('--coverage', action='store_true', help='if the goal is predicting coverage manifold')
    parser.add_argument('--batch_length', type=checkPositiveInt, default=10, dest='batchlength', help='length of mini-batch')
    parser.add_argument('--learning_rate', type=checkPositiveFloat, default=10**(-3), dest='lr', help='learning rate of SGD')
    parser.add_argument('--loss_exponent', type=checkPositiveInt, default=1, dest='loss_exp', help='l1 or l2 loss')

    """
    =========================== Command Line Options for Folder Locations ==============================
    """
    parser.add_argument('--results', type=str, default='Results/', dest='Results', help='path to results directory')
    parser.add_argument('--models', type=str, default='Models/', dest='Models', help='path to models directory')
    parser.add_argument('--data', type=str, default='Data/', dest='Data', help='path to data directory')

    """
    ========================== Command Line Options for Results Generation ===========================
    """
    parser.add_argument('--min_sinr_threshold', type=float, default=-5, dest='MinSINRThr', help='minimum SINR threshold')
    parser.add_argument('--max_sinr_threshold', type=float, default=20, dest='MaxSINRThr', help='maximum SINR threshold')
    parser.add_argument('--num_sinr_threshold', type=checkPositiveInt, default=25, dest='NumSINRThr', help='number of SINR threshold points')
    parser.add_argument('--reruns', type=checkPositiveInt, default=1, dest='TrainReRun', help='number of reruns during training')
    parser.add_argument('--seeds', type=checkPositiveInt, default=1, dest='MaxSeedNum', help='the number of random seeds')

    """ 
    ========================== Command Line Options for the Inverse Problem ============================
    """
    parser.add_argument('--inverse', action='store_true', help='enable inverse problem')
    parser.add_argument('--rand_init_trial', type=checkPositiveInt, default=20, dest='MaxRandInitInv', help='random initialization trials')
    parser.add_argument('--max_deployment', type=checkPositiveInt, default=100, dest='MaxDeployInv', help='maximum allowed brownfield deployments')
    parser.add_argument('--coverage_lower_bound', type=checkFraction, default=0.9, dest='CoverageLBInv', help='minimum coverage threshold to be satisfied')
    parser.add_argument('--inverse_sinr_threshold', type=float, default=0, dest='SINRThrInv', help='SINR threshold for the inverse problem')
    parser.add_argument('--user_fraction_threshold', type=checkFraction, default=0.9, dest='UserFracThrInv', help='minimum fraction of users that must lie above the coverage threshold')

    """ 
    ========================== Command Line Options for the Auxiliary Problem ============================
    """
    parser.add_argument('--auxiliary_sinr_threshold', type=float, default=0, dest='SINRThrAux', help='SINR threshold for simulation in auxiliary problem')
    parser.add_argument('--auxiliary_bs_number', type=float, default=100, dest='NumberBSAux', help='Number of BSs in auxiliary problem')

    args = parser.parse_args()

    """
    ============================== Non-Command Line Parameters ==============================
    """

    args.CountryList = ['India', 'USA', 'Germany', 'Brazil']

    # Folders
    args.DataFolder = args.Data + args.Country + '/'
    args.ModelsFolder = args.Models + args.Country + f'/Shape{args.m_par}/'
    args.ResultsFolder = args.Results + args.Country + f'/Shape{args.m_par}/'
    args.RawResults = args.ResultsFolder + '/Raw/'
    args.InverseResults = args.Results + 'Inverse/'
    args.InverseResultsFolder = args.Results + 'Inverse/' + args.Country + '/'

    # Parameters related to RoI
    args.xIndexMin = int(args.xSteps / 2) - int(int(args.xSteps / 2) * args.MaskRatio)
    args.xIndexMax = int(args.xSteps / 2) + int(int(args.xSteps / 2) * args.MaskRatio)
    args.yIndexMin = int(args.ySteps / 2) - int(int(args.ySteps / 2) * args.MaskRatio)
    args.yIndexMax = int(args.ySteps / 2) + int(int(args.ySteps / 2) * args.MaskRatio)

    args.xLengthMin = (args.xLength / 2) - ((args.xLength / 2) * args.MaskRatio)
    args.xLengthMax = (args.xLength / 2) + ((args.xLength / 2) * args.MaskRatio)
    args.yLengthMin = (args.yLength / 2) - ((args.yLength / 2) * args.MaskRatio)
    args.yLengthMax = (args.yLength / 2) + ((args.yLength / 2) * args.MaskRatio)

    xVec = torch.linspace(args.xLengthMin, args.xLengthMax, steps=int(args.xSteps * args.MaskRatio))
    yVec = torch.linspace(args.yLengthMin, args.yLengthMax, steps=int(args.ySteps * args.MaskRatio))
    args.xMesh, args.yMesh = torch.meshgrid(xVec, yVec)

    # Parameters related to data
    args.EarthR = 6371                                                  # Assume Earth to be a sphere with radius R
    args.lat_div = (180 * args.yLength) / (3.1415 * args.EarthR)        # Corresponds to yLength
    # Longitude division corresponding to xLength is not constant, but depends on latitude

    # Random Variable related to Antenna Gain
    args.antenna_gain_values = torch.tensor([args.main_lobe_t*args.main_lobe_r, args.side_lobe_t*args.main_lobe_r, args.main_lobe_t*args.side_lobe_r, args.side_lobe_t*args.side_lobe_r])
    c_t = args.beamwidth_t/360
    c_r = args.beamwidth_r/360
    antenna_gain_probs = torch.tensor([c_t*c_r, (1-c_t)*c_r, c_t*(1-c_r), (1-c_t)*(1-c_r)])
    args.antenna_gain_dist = torch.distributions.Categorical(antenna_gain_probs)

    """
    ============================== Catch/Change Invalid Inputs ==============================
    """
    if not args.coverage:
        args.NumSINRThr = 1  # Evaluation of rate is SINRThr independent

    if args.Country not in args.CountryList:
        raise ValueError(f'Country Data not available. \n'
                         f'Available countries are {args.CountryList}.')

    return args
