package co.gyeongmin;

import co.gyeongmin.neuralnet.NeuralNetLearning;
import co.gyeongmin.nntest.XORFunction;
import co.gyeongmin.nntest.YX2Function;

public class Main {

    public static void main(String[] args) {
        //xor();

        XORFunction xorFunction = new XORFunction();
        YX2Function yx2Function = new YX2Function(5.0);

        int[][] xorInputSet = { { 0, 0 }, { 0, 1 }, { 1, 0 }, {1 , 1} };

        for (int i = 0; i < xorInputSet.length; i++) {
            System.out.println("XOR Value for (" + xorInputSet[i][0] + ", " + xorInputSet[i][1] + ") = "
                    + xorFunction.xor(xorInputSet[i][0], xorInputSet[i][1]));
        }

        for (double x = -5.0; x < 5.0; x += 0.1) {
            System.out.println("f(" + String.format("%.2f", x) + ") = " + String.format("%.2f", yx2Function.run(x)));
        }
    }

    /*
    *  XOR Test Function
    *
    * | Input Set | Answer |
    * |  0  |  0  |   0    |
    * |  0  |  1  |   1    |
    * |  1  |  0  |   1    |
    * |  1  |  1  |   0    |
    * +-----+-----+--------+
    *
    *
    *
    *  There is 2 factors, learning Rate, momentum Term.
    *
    *  learningRate is the how much should I rely on answers?
    *  And the momentum Term means how much should I rely on my previous answers?
    *
    * */
    public static void xor() {
        NeuralNetLearning neuralNetLearning = new NeuralNetLearning(0.45, 0.9, new int[]{ 2, 2, 2 }, 2, 1);
        double[][] inputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        double[] answers = { 0, 1, 1, 0 };

        int iterate = 1000000;
        while(iterate-- > 0) {
            for (int i = 0; i < 4; i++) {
                neuralNetLearning.forward(inputs[i]);
                neuralNetLearning.backPropagate(new double[]{answers[i]});
            }
        }

        for (int i = 0; i < 4; i++) {
            neuralNetLearning.forward(inputs[i]);
            double[] retVal = neuralNetLearning.getOutput();
            System.out.println("XOR(" + inputs[i][0] + ", " + inputs[i][1] + ") = " + String.format("%.2f", retVal[0]));
        }

        neuralNetLearning.showNetStat();
    }
}
