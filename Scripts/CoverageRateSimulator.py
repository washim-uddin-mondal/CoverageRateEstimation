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
        DistGainMesh = ((xLoc - xMeshBS).pow(2) + (yLoc - yMeshBS).pow(2)).pow(-self.args.alpha / 2)

        MeshOutput = torch.zeros([self.args.NumSINRThr, xSteps, ySteps])

        if self.args.AssocRule == 'Inst':                              # Maximum instantaneous gain based BS association
            for iteration in range(self.args.MaxSimIter):
                FadingMesh = torch.tensor(np.random.gamma(self.args.m_par*np.ones([NumberBS, xSteps, ySteps])))
                GainMesh = FadingMesh * DistGainMesh         # Element wise multiplication
                AssocGain = GainMesh.max(dim=0)[0]           # Gains of associated BSs

                Interference = GainMesh.sum(dim=0) - AssocGain
                SINRMesh = AssocGain/(Interference + self.args.NoiseOverPower)

                for SINRIndex in range(self.args.NumSINRThr):
                    if self.args.coverage:
                        LinSINRThr = 10 ** (self.args.SINRThrVec[SINRIndex] / 10)
                        MeshOutput[SINRIndex, :, :] += ((SINRMesh > LinSINRThr)*1.0 - MeshOutput[SINRIndex, :, :])/(iteration+1)
                    else:
                        MeshOutput[SINRIndex, :, :] += (torch.log2(1+SINRMesh)/self.NormalizingFactor - MeshOutput[SINRIndex, :, :]) / (iteration + 1)

            return MeshOutput

        elif self.args.AssocRule == 'Avg':                             # Maximum average gain based BS association
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
