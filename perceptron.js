function Rand(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min; //The maximum is inclusive and the minimum is inclusive 
}

class Perceptron{
	constructor(){
		this.weightx = Rand(-1,1)
		this.weighty = Rand(-1,1)
		this.weightbias = Rand(-1,1)
		this.bias = 1
		this.lr = 0.5
		this.x = 1;
		this.y = 1;
		this.sum = 0;
		this.expected = 0;
		this.error = 0;
		this.expected = 0;
	}

	positive_train(){
		console.log("positive_train intital", {x:this.weightx, y:this.weighty, bias:this.weightbias})
		this.x = Rand(-255,255)
		this.y = Rand(0,255)

		this.sum = (this.weightx * this.x) + (this.weighty * this.y) + (this.bias * this.weightbias)
		if(this.sum > 0 ){
			this.guess = 1
		}else{
			this.guess = -1
		}

		if(this.y < this.x){
			this.expected = -1;
		}else if(this.y == this.x){
			this.expected = 1;
		}else{
			this.expected = 1;
		}

		this.error = this.expected - this.guess;
		this.weighty += this.y * this.error * this.lr
		this.weightx += this.x * this.error * this.lr
		this.weightbias += this.bias * this.error * this.lr

		console.log("I am positive_train", {x:this.weightx, y:this.weighty, bias:this.weightbias})
		console.log("I am positive_train error", this.error)

	}

	negative_train(){
		console.log("negative_train intital", {x:this.weightx, y:this.weighty, bias:this.weightbias})
		this.x = Rand(-255,255)
		this.y = Rand(-255,0)
		this.sum = (this.weightx * this.x) + (this.weighty * this.y) + (this.bias * this.weightbias)
		if(this.sum > 0 ){
			this.guess = 1
		}else{
			this.guess = -1
		}

		if(this.y < this.x){
			this.expected = -1;
		}else if(this.y == this.x){
			this.expected = 1;
		}else{
			this.expected = 1;
		}

		this.error = this.expected - this.guess;
		this.weighty += this.y * this.error * this.lr
		this.weightx += this.x * this.error * this.lr
		this.weightbias += this.bias * this.error * this.lr

		console.log("I am negative_train", {x:this.weightx, y:this.weighty, bias:this.weightbias})
		console.log("I am negative_train error", this.error)

	}
}

var perceptron = new Perceptron();

while(true == true){
	perceptron.negative_train()
	perceptron.positive_train()
}

/******
**
**	Pereceptron Explanation
**

This perceptron guesses whether any given point is above or below the line y=x. It returns 1 if it thinks its above, and returns -1 if it thinks its below.

The math of this specific perceptron works as follows

The value of the inputs multiplied by their weights and summed. If the sum is greater than zero, the output is 1. If not, the output is -1. This
output is then compared to the expected answer. The output is subtracted from the expected output to produce an error. This error is then multiplied
by the input along with a learning rate and added to the corresponding weight to produce the new weights.

Weights

Each input gets a weight value. There are two parts to a weight, its sign (positive and negative) and its absolute value (magnitude). The sign tells whether something either 
supports or disproves an assumption. A positive sign means the specific input supports the assumption and the negative sign means the specific input
disproves and assumption. The magnitude tells how significant the input is in making the decision. Lets take a task where we have to tell whether a 
police badge is fake as an example. The eixistence of a "fake badge" tag on the badge could be an input that comes in the form of a 0,1 boolean. 
If a tag exists, it would probably work to disprove the assumption that its a real badge and the weight for it would therfore be negative. Also, having
a "fake badge" is a very strong indicator that its fake so the absolute value also would be high. Another input could be shininess or some
measurement of how much light it reflects. Real police badges are metal which are shinier on average then their fake plastic counterparts so the
weight would be positive. However, there are lots of fake metal badges too so the absolute value of the weight would be small as its not a very 
conclusive input.

Inputs

It is useful to think of inputs of having the same two components as errors, absolute value and sign. An input that we expect to support an assumption 
is positive, then it will support the assumption. However, if that same input is negative, it helps to disprove the assumption. The same goes for a 
disproving input. If its positive it will work to disprove to the assumption. But if its negative it will work to support the assumption. It is
useful then, to think of the negative sign as something that reverses meaning. It is important to note that most of the time we wont know whether or 
not an input supports an assumption or disproves an assumption until after its done training. Usually the inputs are normalized to a number between 0 and 1

Bias

The bias exists to make decisions about edge cases. It is determined by the user. The value of the input is always the same but the bias gets its own
weight to be adjusted. This weight should usually be small. In the case of our perceptron, what do we do if the point is on the line y=x? This is where the
bias comes in. It will push our total sum just over the limit so that we count most points on the line y=x as being over the line. In other words, it
helps trigger our activation function in this case. If we wanted to, we can also make the bias can also be negative so that we count all points on 
y=x as below the line and the activation function is not triggered.

Errors

The error is found by subtracting the guessed value from the expected or known value. Yet again, it is useful to think about the error in terms of sign
and absolute value. The absolute value is a measure of how wrong to were. The sign is indicates what direction you need to go to correct it. Lets take
navigating to a point via radar as a simple example. Imagine you are a submarine and you are navigating to a blip on your radar. Lets say you move a small distance forward
towards the blip. When your radar updates, you see that you have gotten closer to the blio but still are a far distance from the blip. Your 
error will be postive becaues you still made progress. The distance in this case is the absolute value of the error so your error will be pretty large
because you are still far away. Now lets say you make a huge jump forward. When your radar updates, you now have passed the blip and it is a small distance
behind you. Your error will be negative indicating that you have to change directions. However, the absolute value will be small because the distance from
the blip is small. The reason you multiply the error by the input has to do with finding the derivative of the cost function (function for finding total error) 
relative to the slope using the least squares method. If you are trying to optimize relative to the y-intercept, you don't multiply the error by the input just
by the learning rate.

Learning rate

The learning rate is a number between 0 to 1 that exists to control how much your weights are adjusted. Without the learning rate, you risk over adjusting
your weights and over shooting the optimum answer.




*/
