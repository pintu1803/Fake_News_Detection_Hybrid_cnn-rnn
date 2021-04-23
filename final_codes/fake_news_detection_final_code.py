

#Created by: Pintu (181CO139) and Akshay Dhayal (181CO105)
#Initial build on: 23/03/2021
#First edit on 22-04-2021
#Objective: To build the hybrid cnn-rnn model to detect the fake news.


#NOTE: This program is the "two in one" program.
# It consists of preprocessing and training codes.

#Important info:
# dataset "fake_news_dataset.csv" should be in same directory as of program.
# http://nlp.stanford.edu/data/glove.6B.zip download the GloVe zip folder.
# extract the zip folder in same directory as of program.
#"glove.6B.100d.txt" and "glove.6B.50d.txt" these files must be in same directory as of program.
#In short, dataset, code and glove should be in same folder
#to run this program successfully, as it is.

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers




#Function for training and validating of model.
def Training_Validating(filename, usable_vocab_size, maxlen_of_text, emb_dim, epochs, batch_size,
                        filename_xtest, filename_ytest, filename_history, filename_model):
    ###load dataset.
    filename=""+filename+".csv"
    dataset = pd.read_csv(filename)

    # remove the missing data.
    dataset = dataset.dropna()

    # split ratio= 8:2
    # 20% is used for validation.
    n = int(0.8 * len(dataset))

    # spliting 'text' and 'label' attributes,
    # for training and testing purposes.
    x = dataset['text'].values[:n]
    y = dataset['label'].values[:n]

    # for validation purpose.
    x_valid = dataset['text'].values[n:]
    y_valid = dataset['label'].values[n:]

    # dividing x,y for training and testing.
    # 20% of entire dataset is used for testing.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1000)

    # types of all splits is 'ndarray'
    # those ending with .values are ndarray.
    # train_test_split output has same type as of input.
    print("Type of dataset =>", type(dataset))
    print("Type of training & testing set=>", type(x))
    print("Type of validation set=>", type(x_valid))

    ###Creating vocabulary.
    tokenizer = Tokenizer(num_words=usable_vocab_size)
    tokenizer.fit_on_texts(x_train)

    # number of unique words in dict.
    print("Number of unique words in dictionary=", len(tokenizer.word_index))

    # replacing words by their index from vocabulary.
    x_train = tokenizer.texts_to_sequences(x_train)
    x_valid = tokenizer.texts_to_sequences(x_valid)
    x_test = tokenizer.texts_to_sequences(x_test)

    # Adding 1 because of  reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    # size of random text in training set.
    print("Length of random text=>")
    print("Length of 4th text of train set=> ", len(x_train[3]))
    print("Length of 14th text of train set=> ", len(x_train[13]))

    # deciding the maximum length of text.
    maxlen = maxlen_of_text

    # padding the texts to maximum length.
    x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
    x_valid = pad_sequences(x_valid, padding='post', maxlen=maxlen)
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

    ###confirm that texts are converted in vector form.
    print("1st text of training set, upto 10 words =>\n", x_test[1][:10])

    ###Create embedding matrix.
    def create_embedding_matrix(filepath, word_index, embedding_dim):
        vocab_size = len(word_index) + 1
        # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        # specify the encoding, o/w it may lead to error.
        with open(filepath, encoding='utf8') as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word]
                    embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
        return embedding_matrix

    # set the size of dense_vector/feature_vector.
    # each word-> dense vector: length=emd_dim
    embedding_dim = emb_dim
    glove_file='glove.6B.'+str(emb_dim)+'d.txt'
    embedding_matrix = create_embedding_matrix(glove_file, tokenizer.word_index, embedding_dim)

    ##Build the hybrid model.
    model = Sequential()
    model.add(
        layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    ##RUN the model, with epochs=10 & batch_size=64
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        validation_data=(x_valid, y_valid),
                        batch_size=batch_size)

    ###Save for 2nd part: testing.
    # save the x_test
    import pickle
    # Syntax of pickle.dump(object_to_dump, filepath)
    with open(filename_xtest, 'wb') as file_pi:
        pickle.dump(x_test, file_pi)

    # save the y_test
    with open(filename_ytest, 'wb') as file_path:
        pickle.dump(y_test, file_path)

    # save the history of the model.
    path = open(filename_history, 'wb')
    pickle.dump(history.history, path)
    path.close()

    # save the trained model.
    model.save(filename_model)

#############################################

def Testing(filename_xtest,filename_ytest,filename_history,filename_model):
    ###Load the saved files.
    # load the x_test.
    import pickle
    file_pi = open(filename_xtest, 'rb')
    x_test = pickle.load(file_pi)
    file_pi.close()

    # laod the y_test.
    file_path = open(filename_ytest, 'rb')
    y_test = pickle.load(file_path)
    file_path.close()

    # load the history
    path = open(filename_history, 'rb')
    history = pickle.load(path)
    path.close()

    # load the model
    from keras.models import load_model
    model = load_model(filename_model)

    # print summary of the model.
    model.summary()
    val_loss, val_acc = model.evaluate(x_test, y_test)

    # printing the evalution result.
    print("Validation accuracy=> %.2f" % (100 * val_acc), "%")
    print("Validation loss=> %.2f" % (100 * val_loss), "%")

    ##Prediction and performance
    def performance(x_test, y_test):
        y_pred = model.predict(x_test)
        # print(y_pred[0])
        y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
        # print(y_pred[0])
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        print("Confusion matrix=\n", cm)
        print("Classification report=\n", cr)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(100*accuracy),"%")  # actual accuracy
        print("Testing Loss: {:.4f}".format(100*loss),"%")  # testing loss

    # call the performance measure function
    performance(x_test, y_test)

    ##Print the loss and accuracy graph.
    import matplotlib.pyplot as plt
    def plot_history(history):
        #already passed the history.history.
        # acc = history['accuracy']
        # val_acc = history['val_accuracy']
        # loss = history['loss']
        # val_loss = history['val_loss']
        # x = range(1, len(acc) + 1)
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(x, acc, 'b', label='Training acc')
        # plt.plot(x, val_acc, 'r', label='Validation acc')
        # plt.title('Training and validation accuracy')
        # plt.show()
        # plt.subplot(1, 2, 2)
        # plt.plot(x, loss, 'b', label='Training loss')
        # plt.plot(x, val_loss, 'r', label='Validation loss')
        # plt.title('Training and validation loss')
        # plt.show()

        plt.plot(history['accuracy'], 'b', label='Training accuracy')
        plt.plot(history['val_accuracy'], 'g', label='Validation accuracy')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history['loss'],'b', label='Training accuracy')
        plt.plot(history['val_loss'],'g', label='Validation accuracy')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training loss', 'Validation loss'], loc='upper left')
        plt.show()

    # call the function using history
    plot_history(history)

############################################

# # main function.
# def main():
#     filename = input("Enter dataset name => ")
#     usable_vocab_size = int(input("Enter size of vocabulary to be used => "))#100000 to 150000
#     maxlen_of_text = int(input("Enter maximum length(size) of the text => "))#300 to 500
#     emb_dim = int(input("Enter length(size) of dense/feature vector =>"))#50 to 300
#     epochs = int(input("Enter number of epochs => "))# 2 to 10
#     batch_size = int(input("Enter batchsize =>"))#32 to 128
#     filename_xtest = input("Enter filename for x_test (to pickle) =>")#xtest1 to xtestN
#     filename_ytest = input("Enter filename for y_test (to pickle) =>")
#     filename_history = input("Enter filename for history (to pickle) =>")
#     filename_model = input("Enter filename for model (to save) =>")
#     Training_Validating(filename, usable_vocab_size, maxlen_of_text, emb_dim, epochs, batch_size,
#                         filename_xtest, filename_ytest, filename_history, filename_model)
#     Testing(filename_xtest, filename_ytest, filename_history, filename_model)
# #############################################
#
# # utility function of main.
# if __name__ == '__main__':
#     main()

##$$$$$$$$$%%%%%%%%%%%^^^^^^^^^&&&&&&&&&&&*************(((((((((((((("FINISH"))))))))))))))@@@@@@@@@@@####################$$$$$$$$$$$$%%%%%%%%%%#####