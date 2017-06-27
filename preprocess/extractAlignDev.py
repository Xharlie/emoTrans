import base64
import csv
import os

filename = "../data/MsCelebV1-Faces-Aligned.part.00.tsv"
outputDir = "../data/MsCelebV1-Faces-Aligned"


def readline(line):
    MID, ImageSearchRank, ImageURL, PageURL, FaceID, FaceRectangle, FaceData = line.split("\t")
    return MID, FaceID, base64.b64decode(FaceData)


def main():
    with open(filename, "r") as f:
        i = 0
        harsh = {};
        for line in f:
            MID, faceID, data = readline(line)
            if not (harsh.has_key(MID)):
                harsh[MID] = 0
            harsh[MID] = harsh[MID] + 1;
            saveDir = os.path.join(outputDir, MID)
            savePath = os.path.join(saveDir, "{}_{}.jpg".format(MID, str(harsh[MID]).zfill(4)))

            if not os.path.exists(saveDir):
                os.mkdir(saveDir)
            with open(savePath, 'wb') as f:
                f.write(data)

            i += 1

            if i % 1000 == 0:
                print("Extracted {} images.".format(i))

if __name__ == "__main__":
    main();