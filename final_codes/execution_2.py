
#Execution-2

from fake_news_detection_final_code import *
# main function.

def main():
    Training_Validating('fake_news_dataset',150000,350,100,10,64,'x_test_2','y_test_2','history_2','model_2')
    Testing('x_test_2','y_test_2','history_2','model_2')
#############################################

# utility function of main.
if __name__ == '__main__':
    main()