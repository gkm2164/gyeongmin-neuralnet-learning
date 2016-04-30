package co.gyeongmin.nntest;

import co.gyeongmin.neuralnet.NeuralNetLearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by gyeongmin on 4/29/16.
 */
public class OpticalRecognitionFunction {

    private String filename;
    private NeuralNetLearning neuralNetLearning;
    private List<TestSet> testSetList;
    public OpticalRecognitionFunction(String filename) throws IOException {
        this.filename = filename;
        this.neuralNetLearning = new NeuralNetLearning(0.45, 0.9, new int[]{256, 256, 256}, 256, 1);
        this.testSetList = new ArrayList<>();

        study();
    }

    void study() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));

        while (true) {
            String[] lines = new String[32];
            String temp;
            temp = br.readLine();
            if (temp == null) {
                break;
            }

            lines[0] = temp;
            for (int i = 1; i < 32; i++) {
                lines[i] = br.readLine();
            }

            temp = br.readLine();
            temp = temp.replaceAll(" ", "");
            int value = Integer.valueOf(temp);

            TestSet ts = new TestSet();

            ts.saveData(lines);
            ts.answer = value;

            testSetList.add(ts);
        }

        for (int i = 0; i < 1000; i++) {
            testSetList.forEach(ts -> {
                neuralNetLearning.forward(ts.data);
                neuralNetLearning.backPropagate(new double[]{(ts.answer / 9.0)});
            });
        }
    }

    public int number(byte[][] data) {

        double[] dblData = new double[16 * 16];

        for (int i = 0; i < data.length; i += 2) {
            for (int j = 0; j < data[i].length; j += 2) {
                dblData[i / 2 * 16 + j / 2] = (data[i][j] + data[i][j + 1] + data[i + 1][j] + data[i + 1][j + 1] - 4 * '0') / 4.0;
            }
        }

        neuralNetLearning.forward(dblData);
        return (int)Math.round(neuralNetLearning.getOutput()[0] * 9.0);
    }

    class TestSet {
        double[] data = new double[16 * 16];
        int answer;

        void saveData(String[] lines) {
            for (int i = 0; i < lines.length; i += 2) {
                for (int j = 0; j < 32; j += 2) {
                    int d1 = lines[i].getBytes()[j] - '0',
                        d2 = lines[i].getBytes()[j + 1] - '0',
                        d3 = lines[i + 1].getBytes()[j] - '0',
                        d4 = lines[i + 1].getBytes()[j + 1] - '0';

                    double value = d1 + d2 + d3 + d4;
                    value /= 4;
                    data[i / 2 * 16 + j / 2] = value;
                }
            }
        }
    }
}
