#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: selectQtlsFromAssoc.py
Author: CrazyHsu @ crazyhsu9527@gmail.com 
Created on: 2020-12-07 11:23:56
Last modified: 2020-12-07 11:23:57
'''

import pandas as pd
import numpy as np
import sys, os
import subprocess, itertools
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from multiprocessing import Pool

import argparse

def getGenePos(geneFile):
    myDict = {}
    with open(geneFile) as f:
        for i in f.readlines():
            infoList = i.strip().split("\t")
            myDict[infoList[3]] = [infoList[0], int(infoList[1]), int(infoList[2]), infoList[3]]
    return myDict

def getCorrelation(plinkLDdata, snps1, snps2, useChunk=False):
    if useChunk:
        # corItem = None
        for chunk in plinkLDdata:
            corItem = chunk.loc[chunk.SNP_A.isin(snps1) & chunk.SNP_B.isin(snps2)]
            if not corItem.empty:
                break
        return corItem
    else:
        corItem = plinkLDdata.loc[plinkLDdata.SNP_A.isin(snps1) & plinkLDdata.SNP_B.isin(snps2)]
        return corItem

def getLdData(ldDict, chrId, useChunk=False):
    if useChunk:
        return pd.read_csv(ldDict[chrId], sep="\s+", usecols=["SNP_A", "SNP_B", "R2"], chunksize=1000000)
    else:
        return ldDict[chrId]

def splitBedAndCalLD(myBed, allChrId, useChunk=False):
    ldDict = {}
    for chrId in allChrId:
        tmpBfilePrefix = "myPlink_{}".format(chrId)
        chrBed = "{}.bed".format(tmpBfilePrefix)
        chrBim = "{}.bim".format(tmpBfilePrefix)
        chrFam = "{}.fam".format(tmpBfilePrefix)
        if not (os.path.exists(chrBed) and os.path.exists(chrBim) and os.path.exists(chrFam)):
            cmd = "plink --bfile {} --chr {} --make-bed --out {} 1>/dev/null".format(myBed, chrId, tmpBfilePrefix)
            subprocess.call(cmd, shell=True, executable="/bin/bash")
            cmd = "plink --bfile {} --r2 --ld-window-r2 0.2 --ld-window-kb 5000 --ld-window 99999 --out plinkLD_{} 1>/dev/null".format(tmpBfilePrefix, chrId)
            subprocess.call(cmd, shell=True, executable="/bin/bash")
        if useChunk:
            ldDict[chrId] = "plinkLD_{}.ld".format(chrId)
            # data = pd.read_csv("plinkLD_{}.ld".format(chrId), sep="\s+", usecols=["SNP_A", "SNP_B", "R2"], chunksize=1000000)
        else:
            data = pd.read_csv("plinkLD_{}.ld".format(chrId), sep="\s+", usecols=["SNP_A", "SNP_B", "R2"])
            ldDict[chrId] = data
        # data = pd.read_csv("plinkLD_{}.ld".format(chrId), sep="\s+", usecols=["SNP_A", "SNP_B"], chunksize=1000000)
        # ldDict[chrId] = data
    return ldDict

def selectQtls(fileList, geneDict, sigPvalue, secondSigPvalue, distLD, qtlDist, snpNumber, ldDict, useChunk=False):
    for myFile in fileList:
        dirname = os.path.dirname(myFile)
        basename = os.path.basename(myFile)
        data = pd.read_csv(myFile, sep="\t")
        data[["CHR", "POS"]] = data.SNP.str.split("_", expand=True)
        data.POS = data.POS.astype(int)
        chrRange = data.groupby("CHR").agg({"POS": ["min", "max"]}).reset_index()
        chrRange.columns = ["chr", "snpMinPos", "snpMaxPos"]

        dataSig = data.loc[data.P <= sigPvalue, :]
        chrList = dataSig.CHR.unique()
        # chrList = ["1", "2", "3"]

        if len(dataSig) > 0:
            chrQtls = {}
            chrQtlBeauty = {}
            for chrId in chrList:
                tmpDataSig = dataSig.loc[dataSig.CHR == chrId, :]
                tmpIndex = [tmpDataSig.index[0]] + list(tmpDataSig.index[tmpDataSig.POS.diff() > distLD]) + [tmpDataSig.index[-1] + 1]
                tmpDataSig.loc[:, "QTL"] = pd.cut(tmpDataSig.index, tmpIndex, right=False, labels=False)
                qtlSummary = tmpDataSig.groupby("QTL").agg({"CHR": lambda x: np.unique(x)[0], "P": ["min", lambda x: int(tmpDataSig.POS[x.idxmin])], "POS": ["min", "max"], "SNP": "count"}).reset_index()
                qtlSummary.columns = ["QTL", "sigP", "peakPos", "chr", "minPos", "maxPos", "snpCount"]
                qtlSummary.peakPos = qtlSummary.peakPos.astype(int)

                if len(qtlSummary) > 1:
                    initN = 0
                    newQtls = [initN]
                    for i in range(len(qtlSummary.index)-1):
                        qtl1 = qtlSummary.loc[qtlSummary.index[i]]
                        qtl2 = qtlSummary.loc[qtlSummary.index[i+1]]
                        if abs(qtl2.peakPos - qtl1.peakPos) <= qtlDist:
                            plinkLDdata = getLdData(ldDict, chrId, useChunk=useChunk)
                            qtl1snps = tmpDataSig.SNP[tmpDataSig.QTL == qtl1.name]
                            qtl2snps = tmpDataSig.SNP[tmpDataSig.QTL == qtl2.name]
                            corItem = getCorrelation(plinkLDdata, qtl1snps, qtl2snps, useChunk=useChunk)
                            # print qtl1snps
                            # print qtl2snps
                            # print corItem
                            # corItem = plinkLDdata.loc[plinkLDdata.SNP_A.isin(qtl1snps) & plinkLDdata.SNP_B.isin(qtl2snps)]
                            if not corItem.empty and (corItem.R2 >= 0.2).any():
                                # print corItem
                                initN += 0
                                newQtls.append(initN)
                            else:
                                initN += 1
                                newQtls.append(initN)
                        else:
                            initN += 1
                            newQtls.append(initN)
                    qtlSummary.loc[:, "newQtl"] = newQtls

                    newQtlSummary = qtlSummary.groupby("newQtl").agg({"chr": lambda x: np.unique(x)[0], "sigP": ["min", lambda x: int(qtlSummary.peakPos[x.idxmin])], "minPos": "min", "maxPos": "max", "snpCount": "sum"}).reset_index()
                    newQtlSummary.columns = ["newQtl", "sigP", "peakPos", "chr", "minPos", "maxPos", "snpCount"]
                else:
                    newQtlSummary = qtlSummary.loc[:, ["newQtl", "sigP", "peakPos", "chr", "minPos", "maxPos", "snpCount"]]
                newQtlSummary.peakPos = newQtlSummary.peakPos.astype(int)

                newQtlLargeN = newQtlSummary[newQtlSummary.snpCount >= snpNumber]
                if len(newQtlLargeN) > 0:
                    newQtlLargeN.loc[:, "threshold"] = sigPvalue
                newQtlLessN = newQtlSummary[newQtlSummary.snpCount < snpNumber]
                if len(newQtlLessN) > 0:
                    newQtlLessN.minPos = newQtlLessN.minPos - distLD/2
                    newQtlLessN.maxPos = newQtlLessN.maxPos - distLD/2

                    newChrRange = newQtlLessN.merge(chrRange, left_on="chr", right_on="chr")
                    newQtlLessN.minPos = np.where(newChrRange.minPos < newChrRange.snpMinPos, newChrRange.snpMinPos,
                                                  newChrRange.minPos)
                    newQtlLessN.maxPos = np.where(newChrRange.maxPos > newChrRange.snpMaxPos, newChrRange.snpMaxPos,
                                                  newChrRange.maxPos)
                    newQtlLessN.loc[:, "snpCount"] = newQtlLessN.apply(
                        lambda x: len(data.loc[(data.P < secondSigPvalue) & (data.CHR == x.chr) & (
                                data.POS >= x.minPos) & (data.POS <= x.maxPos)]), axis=1
                    )
                    newQtlLessN.loc[:, "threshold"] = secondSigPvalue
                    newQtlLessN = newQtlLessN.loc[newQtlLessN.snpCount >= snpNumber]

                mergedNewQtlAdj = pd.concat([newQtlLargeN, newQtlLessN])
                mergedNewQtlAdj = mergedNewQtlAdj.sort_values(by=["chr", "peakPos"])

                newBigQtls = mergedNewQtlAdj.loc[(mergedNewQtlAdj.maxPos-mergedNewQtlAdj.minPos) > 10000000].newQtl
                qtlBeauty = "No"
                if len(newBigQtls) >= 1:
                    newBigQtlsSummary = qtlSummary.loc[qtlSummary.newQtl.isin(newBigQtls)]
                    if len(newBigQtlsSummary) > 1:
                        if (newBigQtlsSummary.groupby("newQtl").agg({"QTL": "count"}).QTL > 3).any() or (
                                newBigQtlsSummary.loc[newBigQtlsSummary.snpCount >= 5].groupby("newQtl").agg(
                                    {"snpCount": "count"}) > 3).any():
                            qtlBeauty = "Bad"
                        else:
                            qtlBeauty = "Beautiful"
                    elif len(newBigQtlsSummary) == 1:
                        qtlBeauty = "Beautiful"
                else:
                    if len(mergedNewQtlAdj) > 2:
                        qtlBeauty = "Bad"
                    elif len(mergedNewQtlAdj) == 1:
                        qtlBeauty = "Beautiful"
                chrQtls[chrId] = mergedNewQtlAdj
                chrQtlBeauty[chrId] = qtlBeauty

            mergedQtls = pd.concat(chrQtls.values(), ignore_index=True)
            mergedQtls = mergedQtls.sort_values(by=["chr", "peakPos"])

            qtlCategory = np.unique(chrQtlBeauty.values())
            print chrQtlBeauty
            if len(qtlCategory) == 1:
                if qtlCategory[0] == "Beautiful":
                    print myFile + " -------> Beautiful!"
                elif qtlCategory[0] == "Bad":
                    print myFile + " -------> Bad!"
                else:
                    print myFile + " -------> No peak!"
            elif len(qtlCategory) == 2:
                if (qtlCategory == np.unique(['Bad', 'Beautiful'])).all():
                    qtlMinPeakSigP = min(-np.log10(mergedQtls.sigP))
                    qtlMaxPeakSigP = max(-np.log10(mergedQtls.sigP))
                    if qtlMaxPeakSigP - qtlMinPeakSigP > 5:
                        print myFile + " -------> Beautiful!"
                    else:
                        print myFile + " -------> Bad!"
                elif (qtlCategory == np.unique(['Beautiful', 'No'])).all():
                    if Counter(chrQtlBeauty.values())["Beautiful"] >= 2:
                        print myFile + " -------> Beautiful with multi peaks!"
                    else:
                        print myFile + " -------> Beautiful!"
                else:
                    print myFile + " -------> Bad and no peak!"
            else:
                qtlMinPeakSigP = min(-np.log10(mergedQtls.sigP))
                qtlMaxPeakSigP = max(-np.log10(mergedQtls.sigP))
                if qtlMaxPeakSigP - qtlMinPeakSigP > 5:
                    print myFile + " -------> Beautiful!"
                else:
                    print myFile + " -------> Bad!"

def selectQtls_bak(fileList, geneDict, sigPvalue, secondSigPvalue, distLD, qtlDist, snpNumber, ldDict):
    for myFile in fileList:
        dirname = os.path.dirname(myFile)
        basename = os.path.basename(myFile)
        data = pd.read_csv(myFile, sep="\t")
        data[["CHR", "POS"]] = data.SNP.str.split("_", expand=True)
        data.POS = data.POS.astype(int)
        chrRange = data.groupby("CHR").agg({"POS": ["min", "max"]}).reset_index()
        chrRange.columns = ["chr", "snpMinPos", "snpMaxPos"]

        dataSig = data.loc[data.P <= sigPvalue, :]
        chrList = dataSig.CHR.unique()

        chrQtls = {}
        chrQtlBeauty = {}
        if len(dataSig) > 0:
            for chrId in chrList:
                tmpDataSig = dataSig.loc[dataSig.CHR == chrId, :]
                tmpIndex = [tmpDataSig.index[0]] + list(tmpDataSig.index[tmpDataSig.POS.diff() > distLD]) + [tmpDataSig.index[-1] + 1]
                tmpDataSig.loc[:, "QTL"] = pd.cut(tmpDataSig.index, tmpIndex, right=False, labels=False)
                qtlSummary = tmpDataSig.groupby("QTL").agg({"CHR": lambda x: np.unique(x)[0], "P": ["min", lambda x: int(tmpDataSig.POS[x.idxmin])], "POS": ["min", "max"], "SNP": "count"}).reset_index()
                qtlSummary.columns = ["QTL", "sigP", "peakPos", "chr", "minPos", "maxPos", "snpCount"]
                qtlSummary.peakPos = qtlSummary.peakPos.astype(int)
                # plinkLDdata = ldDict[chrId]

                if len(qtlSummary) > 1:
                    initN = 0
                    newQtls = [initN]
                    for i in range(len(qtlSummary.index)-1):
                        qtl1 = qtlSummary.loc[qtlSummary.index[i]]
                        qtl2 = qtlSummary.loc[qtlSummary.index[i+1]]
                        if abs(qtl2.peakPos - qtl1.peakPos) <= qtlDist:
                            qtl1snps = tmpDataSig.SNP[tmpDataSig.QTL == qtl1.name]
                            qtl2snps = tmpDataSig.SNP[tmpDataSig.QTL == qtl2.name]
                            corItem = plinkLDdata.loc[plinkLDdata.SNP_A.isin(qtl1snps) & plinkLDdata.SNP_B.isin(qtl2snps)]
                            if not corItem.empty and (corItem.R2 >= 0.2).any():
                                # print corItem
                                initN += 0
                                newQtls.append(initN)
                            else:
                                initN += 1
                                newQtls.append(initN)
                        else:
                            initN += 1
                            newQtls.append(initN)
                    qtlSummary.loc[:, "newQtl"] = newQtls

                    newQtlSummary = qtlSummary.groupby("newQtl").agg({"chr": lambda x: np.unique(x)[0], "sigP": ["min", lambda x: int(qtlSummary.peakPos[x.idxmin])], "minPos": "min", "maxPos": "max", "snpCount": "sum"}).reset_index()
                    newQtlSummary.columns = ["newQtl", "sigP", "peakPos", "chr", "minPos", "maxPos", "snpCount"]
                    # newQtlSummary.peakPos = newQtlSummary.peakPos.astype(int)
                    # newBigQtls = newQtlSummary.loc[(newQtlSummary.maxPos - newQtlSummary.minPos) > 10000000].newQtl
                    # newBigQtlsSummary = qtlSummary.loc[qtlSummary.newQtl.isin(newBigQtls)]
                    # if (newBigQtlsSummary.groupby("newQtl").agg({"QTL": "count"}).QTL > 3).any() or (
                    #         newBigQtlsSummary.loc[newBigQtlsSummary.snpCount >= 5].groupby("newQtl").agg(
                    #                 {"snpCount": "count"}) > 3).any():
                    #     qtlBeauty = "Beauty"
                    # else:
                    #     qtlBeauty = "Bad"
                else:
                    newQtlSummary = qtlSummary.loc[:, ["newQtl", "sigP", "peakPos", "chr", "minPos", "maxPos", "snpCount"]]
                newQtlSummary.peakPos = newQtlSummary.peakPos.astype(int)
                newBigQtls = newQtlSummary.loc[(newQtlSummary.maxPos - newQtlSummary.minPos) > 10000000].newQtl
                newBigQtlsSummary = qtlSummary.loc[qtlSummary.newQtl.isin(newBigQtls)]
                qtlBeauty = None
                if len(newBigQtlsSummary) > 1:
                    if (newBigQtlsSummary.groupby("newQtl").agg({"QTL": "count"}).QTL > 3).any() or (
                            newBigQtlsSummary.loc[newBigQtlsSummary.snpCount >= 5].groupby("newQtl").agg(
                                    {"snpCount": "count"}) > 3).any():
                        qtlBeauty = "Bad"
                    else:
                        qtlBeauty = "Beautiful"
                elif len(newBigQtlsSummary) == 1:
                    qtlBeauty = "Beautiful"

                chrQtls[chrId] = newQtlSummary
                chrQtlBeauty[chrId] = qtlBeauty

            mergedQtls = pd.concat(chrQtls.values(), ignore_index=True)
            mergedQtlsLargeN = mergedQtls[mergedQtls.snpCount >= snpNumber]
            mergedQtlsLargeN.loc[:, "threshold"] = sigPvalue
            mergedQtlsLessN = mergedQtls[mergedQtls.snpCount < snpNumber]

            if len(mergedQtlsLessN) > 0:
                mergedQtlsLessN.minPos = mergedQtlsLessN.minPos - distLD/2
                mergedQtlsLessN.maxPos = mergedQtlsLessN.maxPos + distLD/2

                mergedChrRange = mergedQtlsLessN.merge(chrRange, left_on="chr", right_on="chr")
                mergedQtlsLessN.minPos = np.where(mergedChrRange.minPos < mergedChrRange.snpMinPos, mergedChrRange.snpMinPos,
                                                 mergedChrRange.minPos)
                mergedQtlsLessN.maxPos = np.where(mergedChrRange.maxPos > mergedChrRange.snpMaxPos, mergedChrRange.snpMaxPos,
                                                 mergedChrRange.maxPos)
                mergedQtlsLessN.loc[:, "snpCount"] = mergedQtlsLessN.apply(
                    lambda x: len(data.loc[(data.P < secondSigPvalue) & (data.CHR == x.chr) & (
                            data.POS >= x.minPos) & (data.POS <= x.maxPos)]), axis=1)
                mergedQtlsLessN.loc[:, "threshold"] = secondSigPvalue
                mergedQtlsLessN = mergedQtlsLessN.loc[mergedQtlsLessN.snpCount>=snpNumber]

            mergedQtlsAdj = pd.concat([mergedQtlsLargeN, mergedQtlsLessN])
            mergedQtlsAdj = mergedQtlsAdj.sort_values(by=["chr", "peakPos"])
            qtlMinPeakSigP = min(-np.log10(mergedQtlsAdj.sigP))
            qtlMaxPeakSigP = max(-np.log10(mergedQtlsAdj.sigP))
            if qtlMaxPeakSigP - qtlMinPeakSigP > 5:
                pass


            if len(mergedQtlsAdj) > 0:
                print mergedQtlsAdj


USAGE = 'Filter peaks of association output files in large-scale GWAS studies, such as whole metabolism GWAS or whole gene GWAS'

parser = argparse.ArgumentParser(usage='%(prog)s command [options]', description=USAGE, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-d", dest="dir", type=str, default=".",
                    help="The directory in which association stored.")
parser.add_argument("-suf", dest="suffix", type=str, default="assoc",
                    help="The suffix of the association file, default is assoc.")
parser.add_argument("-bfile", dest="bfile", type=str, default=None, required=True,
                    help="bfile prefix used to run plink. (required)")
parser.add_argument("-g", dest="geneFile", type=str, default=None,
                    help="The file which contains the gene postion with format is: CHR\\tgene_start\\tgene_end\\tgene_name.")
parser.add_argument("-t", dest="threads", type=int, default=2,
                    help="The threads used to run the program, which will be useful when the number of gene or metabolites is very large.")
parser.add_argument("-p1", dest="p1", type=float, default=1e-6,
                    help="The significance threshold for index SNPs.")
parser.add_argument("-p2", dest="p2", type=float, default=1e-5,
                    help="A minor significance threshold for finding SNPs to construct minor qtls.")
parser.add_argument("-r2", dest="r2", type=float, default=0.2,
                    help="LD threshold used to filter SNPs between qtls which are strong related.")
parser.add_argument("-distLD", dest="distLD", type=int, default=800000,
                    help="Physical distance threshold to calculate R2 of SNPs between qtls.")
parser.add_argument("-qtlDist", dest="qtlDist", type=int, default=5000000,
                    help="Physical distance threshold to merge two qtls.")
parser.add_argument("-chunk", dest="chunk", action="store_true",
                    help="Use this option to read plink linkage-disequilibrium(LD) file with chunk. (Default: False)")
# parser.add_argument("-m", dest="mode", type=int, default=1,
#                     help="Select the peaks with four modes, default is 1.\n"
#                          "1: strict mode, which will enable beautiful peaks, and the peak SNP will be located near "
#                          "to the corresponding gene.\n"
#                          "2: less strict mode, which will enable beautiful peaks, and the trans-peak will also be selected besides cis-peaks.\n"
#                          "3: weak mode, which will enable not so beautiful peaks if there are many peaks in association file.\n"
#                          "4: weakest mode, as long as there is peaks in association file pass the filtration, the association file will be kept.")
args = parser.parse_args()

def main():
    fileDir = args.dir

    allChrList = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    # allChrList = ["1", "2", "3"]
    ldDict = splitBedAndCalLD(args.bfile, allChrList, useChunk=args.chunk)

    # fileList = [i for i in os.listdir(fileDir) if i.endswith(args.suffix)]
    # fileList = ["/home/xufeng/xufeng/Projects/selectNicePeakInGwasManhattan/testFiles/salt_all.genes_Zm00001d040060.assoc_tmp"]
    fileList = ["/home/xufeng/xufeng/Projects/selectNicePeakInGwasManhattan/testFiles/salt_all.genes_Zm00001d027737.assoc_tmp"]
    geneDict = getGenePos(args.geneFile)
    # sigPvalue, secondSigPvalue, distLD = args.p1, args.p2, 800000
    # qtlDist = args.qtlDist
    sigPvalue, secondSigPvalue, distLD = 1.0/2140372, 1e-5, 800000
    qtlDist = 5000000
    selectQtls(fileList, geneDict, sigPvalue, secondSigPvalue, distLD, qtlDist, 10, ldDict, useChunk=args.chunk)

if __name__ == '__main__':
    main()
