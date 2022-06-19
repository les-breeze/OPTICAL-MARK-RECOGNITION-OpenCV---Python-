import cv2
import numpy as np
import utils

'''#######Parameters########################'''
widthImg = 650
heightImg = 650
questions = 20
choices = 10
ans = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2, 3, 4]

'''&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'''


'''#CV2 function to read path of a certain image'''
img = cv2.imread('Image1.jpeg')

'''#Processing function'''
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

'''#Finding all Contours'''
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

'''#Finding all rectangles'''
rectCon = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
secondBiggestContour = utils.getCornerPoints(rectCon[1])
thirdBiggestContour = utils.getCornerPoints(rectCon[2])
#print(biggestContour.shape)
# print(courseCodePoints)

if biggestContour.size != 0 and secondBiggestContour.size != 0 and thirdBiggestContour.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, secondBiggestContour, -1, (255, 0, 0), 20)
    cv2.drawContours(imgBiggestContours, thirdBiggestContour, -1, (255, 255, 0), 20)

    biggestContour = utils.reorder(biggestContour)
    courseCodeContour = utils.reorder(secondBiggestContour)

    ''''#The warp perspective'''
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [heightImg, widthImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    pt3 = np.float32(secondBiggestContour)
    pt4 = np.float32([[0, 0], [320, 0], [0, 160], [320, 160]])
    matrix1 = cv2.getPerspectiveTransform(pt3, pt4)
    imgWarpColored1 = cv2.warpPerspective(img, matrix1, (320, 160))
    # cv2.imshow('SECOND BIGGEST CONTOUR', imgWarpColored1)

    '''#APPLYING THRESHOLD'''
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 200, 170, cv2.THRESH_BINARY_INV)[1]

    boxes = utils.splitBoxes(imgThresh)
    # cv2.imshow('Test', boxes[2])
    # print(cv2.countNonZero(boxes[1]), cv2.counNonZero(boxes[2]))

    '''#GETTING NON ZERO PIXEL VALUES FOR EACH BOXES
    AND ALSO ITERATING COLUMNS AND ROWS'''
    myPixelVal = np.zeros((questions, choices))
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if countC == choices: countR += 1; countC = 0
    #print(myPixelVal)

    '''#FINDING INDEX VALUES OF THE MARKINGS'''
    myIndex = []
    for q in range(0, questions):
        arr = myPixelVal[q]
        myIndexVal = np.where(arr == np.amax(arr))
        #print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    # print(myIndex)

    '''#GRADING FUNCTIONS'''
    grading = []
    for q in range(0, questions):
        if ans[q] == myIndex[q]:
            grading.append(1)
        else:
            grading.append(0)
    print("Correct Answers: ", grading)
    score = (sum(grading) / questions) * 100
    print("TOTAL SCORE: ", score)

    '''#DISPLAYING FINAL ANSWERS'''
    imgResult = imgWarpColored.copy()
    imgResult = utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
    imgRawDrawing = np.zeros_like(imgWarpColored)
    imgRawDrawing = utils.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)
    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

    imgRawGrade = np.zeros_like(imgWarpColored1)
    cv2.putText(imgRawGrade, str(int(score))+"%", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
    cv2.imshow("Marks", imgRawGrade)
    #invMatrix1 = cv2.getPerspectiveTransform(pt2, pt1)
    #imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrix1, (widthImg, heightImg))

    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
    #imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

'''IMAGE ARRAY FOR DISPLAY'''
imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
              [imgResult, imgRawDrawing, imgInvWarp, imgFinal])


lables = [['Original', 'Gray', 'Blur', 'Canny'],
          ['Contours', 'BiggestContours', 'Warp view', 'Threshold'],
          ['Result', 'Raw Drawing', 'Inverse Warp', 'Final']]
imgStacked = utils.stackImages(imageArray, 0.3, lables)

#cv2.imshow('DISP', imgFinal)
cv2.imshow('STACKED IMAGES', imgStacked)
cv2.waitKey(0)
