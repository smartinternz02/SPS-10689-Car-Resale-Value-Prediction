package com.org.ml;
import java.io.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.RandomForest;  
import weka.core.Instances;
public class Main {
	
	public static String trainDataPath="";
	public static String testDataPath="";
	public static void main(String args[]) throws Exception{
		String trainFileName="\\\\train1.arff";
		String testFileName="\\\\test1.arff";
		trainDataPath=getCurrentDirectoryPath()+trainFileName;
		//System.out.println(trainDataPath);
		testDataPath=getCurrentDirectoryPath()+testFileName;
		BufferedReader br=new BufferedReader(new FileReader(trainDataPath));
		Instances trainData=new Instances(br);
		BufferedReader brTest=new BufferedReader(new FileReader(testDataPath));
		Instances testData=new Instances(brTest);
		testData.setClassIndex(testData.numAttributes()-1);
		trainData.setClassIndex(trainData.numAttributes()-1);
		br.close();
		brTest.close();
		randomForest(testData,trainData);
	}
	/* 
	 getCurrentDirectoryPath() 
	 -> return type- String
	 -> returns current addresses of the working data-sets.
	 -> implemented to make the code generic, to use the project in any system.
	 */
	private static String getCurrentDirectoryPath() {
		String path=System.getProperty("user.dir")+"\\src\\main\\java\\com\\org\\ml";
		//System.out.println(path);
		return path.replace("\\","\\\\");
	}
	/*
	 *randomForest(Instances i1, Instances i2)
	 *return type- void
	 *complete regressor and the training and testing is done 
	 */
	private static void randomForest(Instances testData, Instances trainData) throws Exception {
		// TODO Auto-generated method stub
		System.out.println("RANDOM FOREST");
		RandomForest rf=new RandomForest();
		rf.setNumIterations(100);
		Evaluation evaluation = new Evaluation(trainData);
		evaluation.crossValidateModel(rf, trainData,10 , new Random(1));
		rf.buildClassifier(testData);
		printResults(evaluation,rf,testData);  
	}
	/*printResults(Evaluation e, RandomForest regressor, Instances instance)
	 * return type- void
	 * results and testing summary is being print
	 */
	private static void printResults(Evaluation eval,RandomForest rf,Instances testData) throws Exception {
		System.out.println("------RESULTS------");
		System.out.println(eval.toSummaryString("",true));
		System.out.println();
		for(int i=0;i<testData.numInstances();i++) {
			String trueLabel;
			trueLabel=testData.instance(i).toString(testData.classIndex());
			double predIndex=rf.classifyInstance(testData.instance(i));
			System.out.println((i+1)+"\t"+trueLabel+"\t\t\t"+predIndex);
		}   
	}
}
