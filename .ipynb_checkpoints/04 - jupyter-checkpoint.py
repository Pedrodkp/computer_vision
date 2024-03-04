import cv2
img = cv2.imread('course material/DATA/00-puppy.jpg')
while True:
    cv2.imshow('Puppy', img)
    #waitKey for one second and check if it is ESC (Escape)
    if cv2.waitKey(1) & 0xFF == 27: #only & extract last 8 bits from waitKey and compare with 'ESC' (27 in ASCII)
        break

    #can be also another key, example 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'): #only & extract last 8 bits from waitKey and compare with 'q'
        break

cv2.destroyAllWindows()