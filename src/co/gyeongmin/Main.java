package co.gyeongmin;

import co.gyeongmin.neuralnet.NeuralNetLearning;

public class Main {

    public static void main(String[] args) {
        /*double[][] inputs = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };
        double[] answer = { 0.0, 1.0, 1.0, 0.0 };*/

        double[] inputs = { 0.0, 1.0, 5.0, 9.0, 10.0 };
        double[] answer = { 1.0, 2.0, 3.0, 4.0, 5.0 };

        NeuralNetLearning neuralNetLearning = new NeuralNetLearning(0.45, 0.9, new int[]{ 10, 10, 10 }, 1, 1);

            int iterate = 100000;
            while (iterate-- > 0) {
                for (int i = 0; i < inputs.length; i++) {
                    double scaleInput = toScale(inputs[i], 0.0, 10.0);
                    double scaleAnswer = toScale(answer[i], 0.0, 5.0);
                    neuralNetLearning.forward(new double[]{ scaleInput });
                    neuralNetLearning.backPropagate(new double[]{ scaleAnswer });
                }
            }

        for (int i = 0; i <= 10; i++) {
            double scaleInput = toScale(i, 0.0, 10.0);
            neuralNetLearning.forward(new double[]{ scaleInput });
            double[] retVal = neuralNetLearning.getOutput();
            //int out = (int)Math.round(retVal[0]);
            //System.out.println("(" + inputs[i][0] + ", " + inputs[i][1] + ")" + "Return: " + out);
            System.out.println("f(" + i + ") = " + fromScale(retVal[0], 0.0, 5.0));
        }

        neuralNetLearning.showNetStat();
    }

    public static double toScale(double value, double min, double max) {
        return (value - min) / (max - min);
    }

    public static double fromScale(double value, double min, double max) {
        return (value * (max - min)) + min;
    }
}
