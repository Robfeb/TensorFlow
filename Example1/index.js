const shape = [4,2];

const data = tf.tensor([4,6,5,9,13,25,1, 57], shape);

// set variables with zeros method
const data2 = tf.variable(tf.zeros([8]));

// print the data
data2.print();

data2.assign(tf.tensor1d([4, 12, 5, 6, 56, 3, 45, 3]));
data2.print();
 
const data3 = tf.tensor1d([4, 6, 5, 9]);
const data4 = tf.tensor1d([5, 4, 23, 45]);

data3.print();
data4.print();

data3.add(data4).print();
data3.mul(data4).print();


// Using a Function Example

function simpleAdd(input1, input2) {
    return tf.tidy(() => {
        const x1 = input1;
        const x2 = input2;
        const y = x1.add(x2);
        return y;
    });
}

// new 1 dimensional tensors/arrays
const data1Arr = tf.tensor1d([4, 6, 5, 9]);
const data2Arr = tf.tensor1d([5, 4, 34, 21]);

const result = simpleAdd(data1Arr,data2Arr);
//printing result
result.print();


//  sequential model
const model = tf.sequential();

model.add(
    tf.layers.simpleRNN({
        // only needed first layer
        inputShape: [20, 4],
        // the number of units or neurons
        units: 20,
        // weight
        recurrentInitializer: 'GlorotNormal',
    })
);
