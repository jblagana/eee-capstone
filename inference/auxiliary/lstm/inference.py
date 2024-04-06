# import pandas as pd

# # Load the data
# data = pd.read_csv('data.csv')
# data_clean = pd.DataFrame()
# data_clean['crowd dynamics'] = data['crowd dynamics']
# data_clean['concealment'] = data['concealment']
# data_clean['loitering'] = data['loitering']
# data_clean['robbery status'] = data['robbery status']


# # Separate the features and the label
# X = data_clean.drop(columns='robbery status')
# y = data_clean['robbery status']


# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# # Build the model
# model = Sequential()
# model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='mse', optimizer='adam')


# # Train the model
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


# # Plot the training and validation loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# # Test the model
# test_loss = model.evaluate(X_test, y_test)
# print('Test Loss:', test_loss)
# # To plot…
