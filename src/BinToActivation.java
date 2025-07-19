import java.io.*;

// A class that converts all Bins to a single activation file

public class  BinToActivation
{
   public static int numRows = 100;
   public static int numColums = 100;
   public static int numImages = 30;
   int[][][] trainingArray;
   int[][] testingArray;
   boolean training;
   boolean testing;
   int totalSets = 6;
   int totalFingers = 5;
   public static String[] letters = {"A", "B", "C", "D", "E", "F"};
   
           public static void main(String[] args)
   {
       BinToActivation processer = new  BinToActivation(true, true);
      processer.fileToArray();
      //System.out.println(processer.testingArray[2][2]);
      processer.arrayToFile();
   }
   public  BinToActivation(boolean training, boolean testing)
   {
      this.training = training;
      this.testing = testing;
   }
   
           public void arrayToFile()
   {
      if (training)
         try
         {
            PrintWriter out = new PrintWriter(new FileWriter("FingersTruthTable"), true);
            for (int set = 1; set < 6; set++)
               for (int finger = 1; finger < 6; finger++)
               {
                  for (int val = 0; val < numRows * numColums; val++)
                     out.println(trainingArray[set][finger - 1][val]);
                  out.println(finger);
                  out.println("Space");
                  System.out.println("Array to TruthTable - " + "Set" + letters[set] + " Finger" + finger);
               }
            out.println("Space");
         }
      catch (FileNotFoundException e) { throw new RuntimeException(e);}
      catch (IOException e) {throw new RuntimeException(e);}
      
      if (testing)
         try
         {
            PrintWriter out = new PrintWriter(new FileWriter("FingersTestCases"), true);
            for (int finger = 1; finger < 6; finger++)
            {
               for (int val = 0; val < numRows * numColums; val++)
                  out.println(testingArray[finger - 1][val]);
               out.println("Space");
               System.out.println("Array to TestCases - " + "Set" + letters[0] + " Finger" + finger);
            }
            out.println("Space");
         }
         catch (FileNotFoundException e) { throw new RuntimeException(e);}
         catch (IOException e) {throw new RuntimeException(e);}
   }
   
           public void fileToArray()
   {
      if (training)
         trainingArray = new int[totalSets][totalFingers][numRows * numColums];
      
      if (testing)
         testingArray = new int[totalFingers][numRows * numColums];
      try
      {
         if (training)
            for (int set = 1; set < 6; set++)
               for (int finger = 1; finger < 6; finger++)
               {
                  FileInputStream fstream = new FileInputStream("processed:Set" + letters[set] + "_F" + finger + ".bin");
                  DataInputStream in = new DataInputStream(fstream);
                  for (int val = 0; val < numRows * numColums; val++)
                     trainingArray[set][finger-1][val] = Integer.valueOf(in.readByte());
                  System.out.println("BinFile to Array - " + "processed:Set" + letters[set] + "_F" + finger + ".bin");
               }
         
         if (testing)
            for (int finger = 1; finger < 6; finger++)
            {
               FileInputStream fstream = new FileInputStream("processed:Set" + letters[0] + "_F" + finger + ".bin");
               DataInputStream in = new DataInputStream(fstream);
               for (int val = 0; val < numRows * numColums; val++)
                  testingArray[finger-1][val] = Integer.valueOf(in.readByte());
               System.out.println("BinFile to Array - " + "processed:Set" + letters[0] + "_F" + finger + ".bin");
            }
      }
      catch (FileNotFoundException e) { throw new RuntimeException(e);}
      catch (IOException e) {throw new RuntimeException(e);}
   }
}