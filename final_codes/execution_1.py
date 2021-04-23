
#Execution-1

from fake_news_detection_final_code import *
# main function.
def main():
    # filename = input("Enter dataset name => ")
    # usable_vocab_size = int(input("Enter size of vocabulary to be used => "))#100000 to 150000
    # maxlen_of_text = int(input("Enter maximum length(size) of the text => "))#300 to 500
    # emb_dim = int(input("Enter length(size) of dense/feature vector =>"))#50 or 100
    # epochs = int(input("Enter number of epochs => "))  # 2 to 10
    # batch_size = int(input("Enter batchsize =>"))#32 to 128
    # filename_xtest = input("Enter filename for x_test (to pickle) =>")#xtest1 to xtestN
    # filename_ytest = input("Enter filename for y_test (to pickle) =>")
    # filename_history = input("Enter filename for history (to pickle) =>")
    # filename_model = input("Enter filename for model (to save) =>")
    # Training_Validating(filename, usable_vocab_size, maxlen_of_text, emb_dim, batch_size,
    #                     filename_xtest, filename_ytest, filename_history, filename_model)
    # Testing(filename_xtest, filename_ytest, filename_history, filename_model)

    Training_Validating('fake_news_dataset',100000,300,50,10,64,'x_test_1','y_test_1','history_1','model_1')
    Testing('x_test_1','y_test_1','history_1','model_1')
#############################################

# utility function of main.
if __name__ == '__main__':
    main()