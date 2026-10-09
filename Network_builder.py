import networks as n

ACTIVATION_FUNCTIONS = {"Relu":n.relu, "Sigmoid":n.sigmoid, "Softmax":n.softmax}
ACTIVATION_DERIVATIVES = {"Relu":n.relu_prime, "Sigmoid":n.sigmoid_prime}

model_name = input("Model Name: ")
n_inputs = int(input("Number of inputs: "))
n_outputs = int(input("Number of outputs: "))
n_hidden = int(input("Number of hidden layers: "))

layer_sizes = []
activations = []
derivatives = []

for i in range(n_hidden):
    print()
    size = int(input(f"Size of hidden layer {i+1}: "))
    layer_sizes.append(size)

    print("Available activation functions:")
    for key in ACTIVATION_DERIVATIVES.keys():
        print(f"- {key}")
    print()
    func = ""
    while(func not in ACTIVATION_DERIVATIVES.keys()):
        func = input(f"Activation function for hidden layer {i+1}: ")
    activations.append(ACTIVATION_FUNCTIONS[func])
    derivatives.append(ACTIVATION_DERIVATIVES[func])
print()

print("Available activation functions:")
for key in ACTIVATION_FUNCTIONS.keys():
    print(f"- {key}")

func = ""
while(func not in ACTIVATION_FUNCTIONS.keys()):
    func = input(f"Activation function for output layer: ")

activations.append(ACTIVATION_FUNCTIONS[func])

print()

model = n.N_Network(layer_sizes, n_inputs, n_outputs, activations, derivatives)
model.save_json(model_name)
if model_name[-5:] != '.json':
    model_name += '.json'
print(f"Model saved to: {model_name}")