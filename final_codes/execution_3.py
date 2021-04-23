
#Execution-3

from fake_news_detection_final_code import *
# main function.

def main():
    Training_Validating('fake_news_dataset',130000,400,100,12,64,'x_test_3','y_test_3','history_3','model_3')
    Testing('x_test_3','y_test_3','history_3','model_3')
#############################################

# utility function of main.
if __name__ == '__main__':
    main()