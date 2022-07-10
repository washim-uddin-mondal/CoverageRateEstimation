""" Simulates coverage and rate manifold over a given rectangular area """
import torch
import numpy as np


class CoverageRateSimulator:
    def __init__(self, args):
        self.args = args
        self.NormalizingFactor = args.RateNormalizingFactor

    def simulate(self, NumberBS, xBS, yBS):
        xSteps = int(self.args.xSteps * self.args.MaskRatio)
        ySteps = int(self.args.ySteps * self.args.MaskRatio)

        # Distance Calculation from BSs
        xMeshBS = xBS.repeat(xSteps * ySteps, 1).transpose(1, 0).reshape(NumberBS, xSteps, ySteps)
        yMeshBS = yBS.repeat(xSteps * ySteps, 1).transpose(1, 0).reshape(NumberBS, xSteps, ySteps)

        xLoc = self.args.xMesh.repeat(NumberBS, 1).reshape(NumberBS, xSteps, ySteps)
        yLoc = self.args.yMesh.repeat(NumberBS, 1).reshape(NumberBS, xSteps, ySteps)
        DistMesh = ((xLoc - xMeshBS).pow(2) + (yLoc - yMeshBS).pow(2)).pow(0.5)

        MeshOutput = torch.zeros([self.args.NumSINRThr, xSteps, ySteps])

        if self.args.real_channel_model:
            # Realistic channel model

            ProbLOSMesh = torch.exp(-DistMesh/self.args.beta_inv)              # Probability that a link is of LOS type

            for iteration in range(self.args.MaxSimIter):
                LinkTypeMesh = (torch.rand([NumberBS, xSteps, ySteps]) > ProbLOSMesh) * 1.0                  # Type 0 = LOS, Type 1 = NLOS
                alphaMesh = self.args.alpha_los * (1 - LinkTypeMesh) + self.args.alpha_nlos * LinkTypeMesh   # Mesh of pathloss exponent
                mMesh = self.args.m_par_los * (1 - LinkTypeMesh) + self.args.m_par_nlos * LinkTypeMesh       # Mesh of fading shape parameter

                DistGainMesh = DistMesh**(-alphaMesh)
                FadingMesh = torch.tensor(np.random.gamma(mMesh))
                ShadowingMesh = torch.exp(self.args.sigma_shadow * torch.randn([NumberBS, xSteps, ySteps]))  # LogNormal
                AntennaGainMesh = self.args.antenna_gain_values[self.args.antenna_gain_dist.sample([NumberBS, xSteps, ySteps])]

                AssocMatrix = DistGainMesh.max(dim=0)[1]                                              # Indices of associated BSs

                GainMeshNoAntenna = ShadowingMesh * FadingMesh * DistGainMesh                         # Element wise multiplication
                AssocGainNoAntenna = torch.gather(GainMeshNoAntenna, 0, AssocMatrix.unsqueeze(0))     # Gains of associated BSs with No Antenna Gain
                AssocGainNoAntenna = AssocGainNoAntenna.squeeze()

                GainMesh = AntennaGainMesh * GainMeshNoAntenna
                AssocGain = torch.gather(GainMesh, 0, AssocMatrix.unsqueeze(0))                       # Total gains of associated BSs
                AssocGain = AssocGain.squeeze()

                Interference = GainMesh.sum(dim=0) - AssocGain
                SINRMesh = self.args.antenna_gain_values[0] * AssocGainNoAntenna / (Interference + self.args.NoiseOverPower)

                for SINRIndex in range(self.args.NumSINRThr):
                    if self.args.coverage:
                        LinSINRThr = 10 ** (self.args.SINRThrVec[SINRIndex] / 10)
                        MeshOutput[SINRIndex, :, :] += ((SINRMesh > LinSINRThr) * 1.0 - MeshOutput[SINRIndex, :, :]) / (iteration + 1)
                    else:
                        MeshOutput[SINRIndex, :, :] += (torch.log2(1 + SINRMesh) / self.NormalizingFactor - MeshOutput[SINRIndex, :, :]) / (iteration + 1)

        else:
            # Rayleigh faded channel with no consideration of shadowing, antenna gain or LOS/NLOS path
            DistGainMesh = DistMesh.pow(-self.args.alpha)

            for iteration in range(self.args.MaxSimIter):
                FadingMesh = torch.tensor(np.random.gamma(self.args.m_par*np.ones([NumberBS, xSteps, ySteps])))
                GainMesh = FadingMesh * DistGainMesh                              # Element wise multiplication
                AssocMatrix = DistGainMesh.max(dim=0)[1]                          # Indices of associated BSs
                AssocGain = torch.gather(GainMesh, 0, AssocMatrix.unsqueeze(0))   # Gains of associated BSs
                AssocGain = AssocGain.squeeze()

                Interference = GainMesh.sum(dim=0) - AssocGain
                SINRMesh = AssocGain / (Interference + self.args.NoiseOverPower)

                for SINRIndex in range(self.args.NumSINRThr):
                    if self.args.coverage:
                        LinSINRThr = 10 ** (self.args.SINRThrVec[SINRIndex] / 10)
                        MeshOutput[SINRIndex, :, :] += ((SINRMesh > LinSINRThr) * 1.0 - MeshOutput[SINRIndex, :, :]) / (iteration + 1)
                    else:
                        MeshOutput[SINRIndex, :, :] += (torch.log2(1 + SINRMesh) / self.NormalizingFactor - MeshOutput[SINRIndex, :, :]) / (iteration + 1)

        return MeshOutput
