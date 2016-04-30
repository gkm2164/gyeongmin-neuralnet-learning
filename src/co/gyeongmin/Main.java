package co.gyeongmin;

import co.gyeongmin.neuralnet.NeuralNetLearning;
import co.gyeongmin.nntest.OpticalRecognitionFunction;
import co.gyeongmin.nntest.XORFunction;
import co.gyeongmin.nntest.YX2Function;

import java.io.IOException;

public class Main {

    static String[] x = {
            "00000000000001100111100000000000",
            "00000000000111111111111111000000",
            "00000000011111111111111111110000",
            "00000000011111111111111111110000",
            "00000000011111111101000001100000",
            "00000000011111110000000000000000",
            "00000000111100000000000000000000",
            "00000001111100000000000000000000",
            "00000001111100011110000000000000",
            "00000001111100011111000000000000",
            "00000001111111111111111000000000",
            "00000001111111111111111000000000",
            "00000001111111111111111110000000",
            "00000001111111111111111100000000",
            "00000001111111100011111110000000",
            "00000001111110000001111110000000",
            "00000001111100000000111110000000",
            "00000001111000000000111110000000",
            "00000000000000000000001111000000",
            "00000000000000000000001111000000",
            "00000000000000000000011110000000",
            "00000000000000000000011110000000",
            "00000000000000000000111110000000",
            "00000000000000000001111100000000",
            "00000000001110000001111100000000",
            "00000000001110000011111100000000",
            "00000000001111101111111000000000",
            "00000000011111111111100000000000",
            "00000000011111111111000000000000",
            "00000000011111111110000000000000",
            "00000000001111111000000000000000",
            "00000000000010000000000000000000",
    };

    public static void main(String[] args) throws IOException {
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

        OpticalRecognitionFunction orf = new OpticalRecognitionFunction("optical.txt");

        byte[][] data = new byte[32][];

        for (int i = 0; i < 32; i++) {
            data[i] = new byte[32];
            System.arraycopy(x[i].getBytes(), 0, data[i], 0, 32);
        }

        /*char[][] d16 = new char[16][16];

        for (int i = 0; i < 32; i += 2) {
            for (int j = 0; j < 32; j += 2) {
                d16[i / 2][j / 2] = (char)((char)Math.round(((int)data[i][j] + data[i][j + 1] + data[i + 1][j] + data[i + 1][j + 1] - 4 * '0') / 4) + '0');
                System.out.print(d16[i/2][j/2]);
            }
            System.out.println();
        }*/

        System.out.println("인식된 값 " + orf.number(data));

    }
}
