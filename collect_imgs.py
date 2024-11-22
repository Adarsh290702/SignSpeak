import os
import cv2

# Use absolute path to avoid any issues
DATA_DIR = os.path.abspath('/Users/adarshupadhyay/Documents/Final Project/Sign Lanuage Detection using Landmarking/Data')

# Debugging print
print("DATA_DIR is:", DATA_DIR)

# Check if the directory exists, if not create it
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR)
        print(f'Directory {DATA_DIR} created successfully.')
    except Exception as e:
        print(f'Error creating directory: {e}')
else:
    print(f'Directory {DATA_DIR} already exists.')

number_of_classes = 36
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        try:
            os.makedirs(class_dir)
            print(f'Directory {class_dir} created for class {j}.')
        except Exception as e:
            print(f'Error creating class directory {class_dir}: {e}')
            continue  # Skip this class if there's an issue

    print(f'Collecting data for class {j}')

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'PRESS Q TO START', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(36) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(36)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()