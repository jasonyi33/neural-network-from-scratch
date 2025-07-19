import java.io.*;
import java.lang.Math;
import java.util.Scanner;
import java.text.DecimalFormat;


/**
 * This class implements backpropagation to optimize weights and minimize the output error function.
 * This feed forward N-layer network is organized with N activation layers.
 * This class may train or run with random or manual weights. It can also load and save weights.
 * This class may also train within an error threshold, iteration threshold, and within a certain number of test cases.
 * This class reads the config parameters from a control file.
 * Sigmoid is used as the activation function.
 *
 * Methods:
 * public void setConfigParams(String con)
 * public void echoConfigParams()
 * public void allocateArrays()
 * public void populateArrays()
 * public void saveWeights()
 * public void loadWeights()
 * public double randomize(double high, double low)
 * public double sigmoid(double input)
 * public double sigmoidDeriv(double input)
 * public double activationFunction(double input)
 * public double activationDeriv(double in)
 * public void trainingForwardPass(int pass)
 * public void runForwardPass(int tests)
 * public void trainNetwork()
 * public void runNetwork()
 * public void reportResults()
 * public static void main(String[] args)
 *
 * @author Jason Yi
 * @version 5/3/2024
*/


public class Network
{
   public static final String DEFAULT_CONFIG = "controlFile.txt";
   public static final Double DEFAULT_WEIGHT_VALUE = 0.0;
   public static final Integer INPUTS_LOC = 0;
   public static final Integer HIDDENS_ONE_LOC = 1;
   public static final Integer SINGLE_OFFSET = 1;
   public String configFile;
   public String testCaseFile;
   public String inputFile;
   public String weightFile;
   public Scanner configScanner;
   public Scanner inputScanner;
   public Scanner testCaseScanner;
   public Scanner weightScanner;
   public long elapsed;
   public int[] numActs;                  // array of number of activations nodes in the different layers
   public int outputLocation;             // number that represents the location of the output layer
   public int numTestCases;               // number of test cases
   public int numTotalLayers;             // number of total layers
   public int numConLayers;               // number of connectivity layers
   public int maxTotalNodes;              // number of max nodes in any layer
   public int maxHiddenOutputNodes;       // number of max nodes in the layers past the input layer
   public int maxIter;                    // maximum number of iterations that training set will run
   public int iterTracker;                // tracker for number of iterations
   public boolean training;               // if network is running - false; if network is training - true
   public double lambda;                  // magnitude of weight change for optimization
   public double lowRandWeight;           // lower bound of randomized weight value
   public double highRandWeight;          // higher bound of randomized weight value
   public double errorThresh;             // maximum error to be considered accurate
   public double errorTracker;            // tracker for the average error throughout the network;
   public double tempError;               // tracker for total error in training
   public double[][] acts;                // array of activation nodes
   public double[][] finalOutputs;        // calculated output values for each test case
   public double[][] inputDataset;        // array of input dataset
   public double[][] outputDataset;       // array of expected output dataset
   public double[][][] weights;           // array of consolidated weights
   public double[][] thetas;
   public double[][] psis;
   public int weightsMode;                // if 1 - load from file, if 2 - randomize, if 3 - preset weights
   public boolean saving;                 // true if weights are to be saved - false otherwise
   public int keepAliveTracker;           // tracker for number of iterations between messages
   private static final DecimalFormat df = new DecimalFormat("0.000");

   /**
    * Constructor
    */
   public Network()
   {
   }

   /**
    * Initializes configuration for the network through a control file
    */
   public void setConfigParams(String con)
   {
      configFile = "/Users/jasonyi/Desktop/IdeaProjects/Proj2_AB1_Network/out/production/Proj2_AB1_Network/" + con;
      try
      {
         configScanner = new Scanner(new File(configFile));
      }
      catch (FileNotFoundException e)
      {
         System.out.println("Config file could not be found");
      }

      numTotalLayers = Integer.parseInt(configScanner.nextLine());
      numTestCases = Integer.parseInt(configScanner.nextLine());

      numConLayers = numTotalLayers - 1;
      maxTotalNodes = 0;
      maxHiddenOutputNodes = 0;
      numActs = new int[numTotalLayers];
      outputLocation = numTotalLayers - 1;

      for (int n = 0; n < numTotalLayers; n++)
      {
         numActs[n] = Integer.parseInt(configScanner.nextLine());
         maxTotalNodes = Math.max(numActs[n],maxTotalNodes);
         if (n > 0)
         {
            maxHiddenOutputNodes = Math.max(numActs[n],maxHiddenOutputNodes);
         }
      }

      maxIter = Integer.parseInt(configScanner.nextLine());
      lambda = Double.parseDouble(configScanner.nextLine());
      lowRandWeight = Double.parseDouble(configScanner.nextLine());
      highRandWeight = Double.parseDouble(configScanner.nextLine());
      errorThresh = Double.parseDouble(configScanner.nextLine());

      training = Boolean.parseBoolean(configScanner.nextLine());
      weightsMode = Integer.parseInt(configScanner.nextLine());
      saving = Boolean.parseBoolean(configScanner.nextLine());
      keepAliveTracker = Integer.parseInt(configScanner.nextLine());

      //testCaseFile = configScanner.nextLine();
      inputFile = configScanner.nextLine();
      weightFile = configScanner.nextLine();

      return;
   } //public void setConfigParams()

   /**
    * Prints the configuration parameters of the function for the user
    * If training, prints the runtime training parameters and network structure
    * If running, only prints the network structure
    */
   public void echoConfigParams()
   {
      System.out.println();
      System.out.println("Config File: " + configFile);
      System.out.println("Input File: " + inputFile);
      if (training)
      {
         System.out.println("Testcase File: " + testCaseFile);
      }

      if (weightsMode == 1)
      {
         System.out.println("Weights will be loaded from a file: " + weightFile);
      }
      else if (weightsMode == 2)
      {
         System.out.println("Weights will be randomized");
      }
      else
      {
         System.out.println("Weights are all set to a constant (0.0)");
      }

      System.out.println();
      System.out.print("Network: ");
      for (int n = 0; n < numTotalLayers; n++)
      {
         System.out.print(numActs[n] + "-");
      }
      System.out.println();
      System.out.println("Training State: " + training);
      System.out.println("Saving State: " + saving);
      System.out.println();

      if (training)
      {
         System.out.println("Lambda: " + lambda);
         System.out.println("Max Iterations: " + maxIter);
         System.out.println("Error Threshold: " + errorThresh);
         System.out.println("Random Number Range: " + lowRandWeight + " - " + highRandWeight);
         System.out.println("Keep Alive Value:" + keepAliveTracker);
      }

      if (saving)
      {
         System.out.println("Weights will be saved to file: " + weightFile);
      }
      else
      {
         System.out.println("Weights will not be saved");
      }
      return;
   } //public void echoConfigParams()

   /**
    * Allocates memory for each array declared in the global variables section.
    * Only allocates memory for thetas and psis and output dataset if training
    */
   public void allocateArrays()
   {
      inputDataset = new double[numTestCases][numActs[INPUTS_LOC]];
      finalOutputs = new double[numTestCases][numActs[outputLocation]];

      weights = new double[numConLayers + SINGLE_OFFSET][][];
      for (int n = 1; n < numConLayers + SINGLE_OFFSET; n++)
      {
         weights[n] = new double[numActs[n-1]][numActs[n]];
      }

      acts = new double[numTotalLayers][maxTotalNodes];

      if (training)
      {
         outputDataset = new double[numTestCases][numActs[outputLocation]];
         thetas = new double[numConLayers + SINGLE_OFFSET][maxHiddenOutputNodes];
         psis = new double[numTotalLayers][maxTotalNodes];
      }

      return;
   } //public void allocateArrays()

   /**
    * Loads the test cases (input and output datasets) from a configuration file
    */
   public void populateArrays()
   {
      switch (weightsMode)
      {
         case 1:              //loads weights from a file as directed by the controlFile
         {
            loadWeights();
            break;
         }

         case 2:              //sets all weights to a random number as determined by the range given in the controlFile
         {
            for (int n = HIDDENS_ONE_LOC; n < numTotalLayers; n++)
            {
               for (int j = 0; j < numActs[n-1]; j++)
               {
                  for (int k = 0; k < numActs[n]; k++)
                  {
                     weights[n][j][k] = randomize(highRandWeight, lowRandWeight);
                  }
               }
            } // for (int n = HIDDENS_ONE_LOC; n < numTotalLayers; n++)
            break;
         } //case 2:

         case 3:              //sets all weights to a constant
         {
            for (int n = HIDDENS_ONE_LOC; n < numTotalLayers; n++)
            {
               for (int j = 0; j < numActs[n-1]; j++)
               {
                  for (int k = 0; k < numActs[n]; k++)
                  {
                     weights[n][j][k] = DEFAULT_WEIGHT_VALUE;
                  }
               }
            } // for (int n = HIDDENS_ONE_LOC; n < numTotalLayers; n++)
            break;
         } //case 3:
      } //switch (weightsMode)

      try
      {
         inputScanner = new Scanner(new File(inputFile));
      }
      catch (FileNotFoundException e)
      {
         System.out.println("Input file could not be found");
      }

      /**
      try
      {
         testCaseScanner = new Scanner(new File(testCaseFile));
      }
      catch (FileNotFoundException e)
      {
         System.out.println("Test case file could not be found");
      }
       */

      inputScanner.nextLine();
      inputScanner.nextLine();
      String temp;

      try
      {
         for (int k = 0; k < numTestCases; k++)
         {
            for (int i = 0; i < numActs[INPUTS_LOC]; i++)
            {
               temp = inputScanner.nextLine();
               inputDataset[k][i] = Double.parseDouble(temp.substring(0, temp.indexOf("=")));
            }
         }
      } //try

      catch (Exception e)
      {
         throw new RuntimeException("Input dataset too big");
      }


      if (training)
      {
         inputScanner.nextLine();

         try
         {
            for (int i = 0; i < numActs[outputLocation]; i++)
            {
               for (int k = 0; k < numTestCases; k++)
               {
                  temp = inputScanner.nextLine();
                  outputDataset[k][i] = Double.parseDouble(temp.substring(0, temp.indexOf("=")));
               }
            }
         } //try

         catch (Exception e)
         {
            throw new RuntimeException("Output dataset too big");
         }
      }
      inputScanner.close();
      return;
   } //public void populateArrays()


   /**
    * Saves the input to hidden and from the hidden to output weights into a text file called Weights.txt
    * The weights are saved where each line contains a single weight, with a single line separating weights of
    * different layers
    */
   public void saveWeights()
   {
      try
      {
         PrintWriter out = new PrintWriter(new File(weightFile));
         out.println("Network: ");
         for (int n = 0; n < numTotalLayers; n++)
         {
            out.print(numActs[n] + "-");
         }
         out.println();

         for (int n = HIDDENS_ONE_LOC; n < numTotalLayers; n++)
         {
            for (int k = 0; k < numActs[n-1]; k++)
            {
               for (int j = 0; j < numActs[n]; j++)
               {
                  out.println(String.format("%.17f", weights[n][k][j]));
               }
               out.println();
            }
         } // for (int n = HIDDENS_ONE_LOC; n < numTotalLayers; n++)

         out.close();
         System.out.println();
         System.out.println("Weights have saved");                       //indicates completion

      } //try
      catch (IOException e)
      {
         System.out.println();
         System.out.println("Weights could not be saved");
      }
   } //public void saveWeights()

   /**
    * Loads the weights from a text file named Weights.txt
    * Parses each line as a double for a weight and updates the network weight arrays
    * Prints "Weights have loaded" upon completion
    */
   public void loadWeights()
   {
      try
      {
         weightScanner = new Scanner(new File(weightFile));

         weightScanner.nextLine();
         weightScanner.nextLine();

         for (int n = HIDDENS_ONE_LOC; n < numTotalLayers; n++)
         {
            for (int k = 0; k < numActs[n-1]; k++)
            {
               for (int j = 0; j < numActs[n]; j++)
               {
                  weights[n][k][j] = Double.parseDouble(weightScanner.nextLine());
               }
               weightScanner.nextLine();
            }
         }

         weightScanner.close();
         System.out.println();
         System.out.println("Weights have loaded");                        //indicates completion
      } //try

      catch(FileNotFoundException e)
      {
         System.out.println();
         System.out.println("Weights could not be loaded");
      }
   } //public void loadWeights()

   /**
    * Calculates a random weight value using the high and low variables
    *
    * @param high - higher bound of randomized weight value calculation
    * @param low - lower bound of randomized weight value calculation
    * @return randomized value for weight
    */
   public double randomize(double high, double low)
   {
      return Math.random() * (high - low) + low;
   }

   /**
    * Sigmoid calculation
    *
    * @param input: input into the sigmoid
    * @return the output of the sigmoid
    */
   public double sigmoid(double input)
   {
      input =  1.0 / (1.0 + Math.exp(-input));
      return input;
   }

   /**
    * Sigmoid derivative calculation
    *
    * @param input: input into the sigmoid derivative
    * @return the output of the sigmoid derivative
    */
   public double sigmoidDeriv(double input)
   {
      input = sigmoid(input);
      return input * (1.0 - input);
   }

   /**
    * Activation function
    *
    * @param input: input into the activation function
    * @return the output of the activation function
    */
   public double activationFunction(double input)
   {
      return sigmoid(input);
   }

   /**
    * Calculates the derivative of the activation function, in this case a sigmoid
    *
    * @param in: value of the activation function
    * @return the derivative of the input activation
    */
   public double activationDeriv(double in)
   {
      return sigmoidDeriv(in);
   }


   /**
    * Performs the forward pass for training hidden and output activations.
    * Updates the activations based on dot product of weights and input dataset.
    * Also applies the activation function
    */
   public void trainingForwardPass(int pass)
   {
      for (int n = HIDDENS_ONE_LOC; n < outputLocation; n++)
      {
         for (int k = 0; k < numActs[n]; k++)
         {
            thetas[n][k] = 0.0;                                              //clearing saved theta
            for (int j = 0; j < numActs[n-1]; j++)
            {
               thetas[n][k] += weights[n][j][k] * acts[n-1][j];              //applying dot product
            }
            acts[n][k] = activationFunction(thetas[n][k]);                   //applying activation function
         } //for (int k = 0; k < numActs[n]; k++)
      } //for (int n = HIDDENS_ONE_LOC; n < outputLocation; n++)

      double tempTheta;
      int n = outputLocation;
      for (int k = 0; k < numActs[n]; k++)                                    //finding hidden to output
      {
         tempTheta = 0.0;                                                     //clearing temp theta
         for (int j = 0; j < numActs[n-1]; j++)
         {
            tempTheta += weights[n][j][k] * acts[n-1][j];                     //applying dot product
         }
         acts[n][k] = activationFunction(tempTheta);                          //applying activation function
         psis[n][k] = (outputDataset[pass][k] - acts[n][k]) * (activationDeriv(tempTheta));
      } // for (int k = 0; k < numActs[n]; k++)
      return;
   } // public void trainingForwardPass(int pass)

   /**
    * Performs a single forward pass for running
    */
   public void runForwardPass()
   {
      for (int n = HIDDENS_ONE_LOC; n < numTotalLayers; n++)
      {
         for (int k = 0; k < numActs[n]; k++)
         {
            thetas[n][k] = 0.0;                                                 //clearing temp theta
            for (int m = 0; m < numActs[n-1]; m++)
            {
               thetas[n][k] += weights[n][m][k] * acts[n-1][m];                 //applying dot product
            }
            acts[n][k] = activationFunction(thetas[n][k]);                      //applying activation function
         } // for (int k = 0; k < numActHiddenOne; k++)
      } // for (int n = HIDDENS_ONE_LOC; n < numTotalLayers; n++)

      return;
   } // public void runForwardPass()

   /**
    * Initializes training for the network. Until an error threshold or number of iterations is met,
    * the network makes a forward pass, calculates the error, stores that, and then applies that to
    * adjust its weights.
    */
   public void trainNetwork()
   {
      iterTracker = 0;
      boolean trainingBool = false;
      double omegas;
      long startTime = System.currentTimeMillis();
      System.out.println();

      while (!trainingBool)
      {
         tempError = 0.0;
         for (int k = 0; k < numTestCases; k++)
         {
            for (int j = 0; j < numActs[INPUTS_LOC]; j++)
            {
               acts[INPUTS_LOC][j] = inputDataset[k][j];
            }

            trainingForwardPass(k);

            for (int n = outputLocation-1; n > HIDDENS_ONE_LOC; n--)
            {
               for (int j = 0; j < numActs[n]; j++)
               {
                  omegas = 0.0;
                  for (int m = 0; m < numActs[n+1]; m++)
                  {
                     omegas += psis[n+1][m] * weights[n+1][j][m];
                     weights[n+1][j][m] += lambda * acts[n][j] * psis[n+1][m];
                  }
                  psis[n][j] = omegas * activationDeriv(thetas[n][j]);
               } // for (int j = 0; j < numActs[n]; j++)
            } // for (int n = outputLocation-1; n > HIDDENS_ONE_LOC; n--)

            int n = HIDDENS_ONE_LOC;
            for (int j = 0; j < numActs[n]; j++)
            {
               omegas = 0.0;
               for (int m = 0; m < numActs[n+1]; m++)
               {
                  omegas += psis[n+1][m] * weights[n+1][j][m];
                  weights[n+1][j][m] += lambda * acts[n][j] * psis[n+1][m];
               }
               psis[n][j] = omegas * activationDeriv(thetas[n][j]);

               for (int l = 0; l < numActs[INPUTS_LOC]; l++)
               {
                  weights[n][l][j] += lambda * acts[n-1][l] * psis[n][j];
               }
            } //for (int j = 0; j < numActs[n]; j++)

            runForwardPass();

            n = outputLocation;
            for (int j = 0; j < numActs[n]; j++)
            {
               finalOutputs[k][j] = acts[n][j];
               tempError += 0.5 * (outputDataset[k][j]-acts[n][j]) * (outputDataset[k][j]-acts[n][j]);
            }
         } // for (int k = 0; k < numTestCases; k++)

         iterTracker++;
         errorTracker = (tempError/ (double) numTestCases);
         trainingBool = ((errorTracker < errorThresh) || (iterTracker >= maxIter));

         if (keepAliveTracker != 0 && iterTracker % keepAliveTracker == 0)
         {
            System.out.printf("Iteration %d, Error = %f\n", iterTracker, errorTracker);
         }

      } // while (!trainingBool)

      elapsed = System.currentTimeMillis() - startTime;

   } // public void trainNetwork()


   /**
    * Initializes the running of the network
    * Executes a forward pass of the network for all inputs
    */
   public void runNetwork()
   {
      for (int tests = 0; tests < numTestCases; tests++)
      {
         for (int j = 0; j < numActs[INPUTS_LOC]; j++)
         {
            acts[INPUTS_LOC][j] = inputDataset[tests][j];
         }

         double tempTheta;

         for (int n = HIDDENS_ONE_LOC; n < outputLocation; n++)
         {
            for (int k = 0; k < numActs[n]; k++)
            {
               tempTheta = 0.0;                                                 //clearing temp theta
               for (int m = 0; m < numActs[n-1]; m++)
               {
                  tempTheta += weights[n][m][k] * acts[n-1][m];                 //applying dot product
               }
               acts[n][k] = activationFunction(tempTheta);                      //applying activation function
            } // for (int k = 0; k < numActHiddenOne; k++)
         } // for (int n = HIDDENS_ONE_LOC; n < outputLocation; n++)

         int n = outputLocation;
         for (int i = 0; i < numActs[n]; i++)
         {
            tempTheta = 0.0;
            for (int j = 0; j < numActs[n-1]; j++)
            {
               tempTheta += weights[n][j][i] * acts[n-1][j];
            }
            acts[n][i] = activationFunction(tempTheta);
            finalOutputs[tests][i] = acts[n][i];
         } // for (int i = 0; i < numActs[n]; i++)
      } //for (int tests = 0; tests < numTestCases; tests++)
   } //public void runNetwork()

   /**
    * Reports results for the training or running of the network.
    * If training, reports the reason for ending, the number of iterations reached, and the calculated error
    * If running, reports the calculated error
    * For both, prints out the truth table showing the inputs, expected output, and network-calculated outputs
    */
   public void reportResults()
   {
      System.out.println();

      if (training)
      {
         System.out.print("REASON FOR END: ");

         if ((iterTracker >= maxIter) && (errorTracker <= errorThresh))
         {
            System.out.print("MAXIMUM ITERATIONS & ERROR THRESHOLD REACHED- " +  maxIter + " - " + errorThresh);
         }

         else if (iterTracker >= maxIter)
         {
            System.out.print("MAXIMUM ITERATIONS REACHED- " +  maxIter);
         }

         else
         {
            System.out.print("ERROR THRESHOLD REACHED- " + errorThresh);
         }

         System.out.println();
         System.out.println("Iterations reached: " + iterTracker);
         System.out.println("Error reached: " + errorTracker);
         System.out.println("Time taken: " + elapsed + " ms");
      }  //if (training)


      if (training)
      {
         /**
         System.out.println("\nTable:        Input   Expected Output        Actual Output");

         for (int k = 0; k < numTestCases; k++)
         {
            System.out.print("Test Case " + (k + 1) + ": |");

            for (int j = 0; j < numActs[INPUTS_LOC]; j++)
            {
               System.out.print(" " + String.format("%.0f", inputDataset[k][j]) + " ");
            }
            System.out.print("| - |");

            for (int j = 0; j < numActs[outputLocation]; j++)
            {
               System.out.print(" " + String.format("%.0f", outputDataset[k][j]) + " ");
            }
            System.out.print("| - |");

            for (int j = 0; j < numActs[outputLocation]; j++)
            {
               System.out.print(" " + String.format("%.17f", finalOutputs[k][j]) + " ");
            }
            System.out.println("|");
         } // for (int k = 0; k < numTestCases; k++)
          */

         System.out.println("Actual Values:");

         for (int eoindex = 0; eoindex < numActs[outputLocation]; eoindex++)
         {
            for (int tindex = 0; tindex < numTestCases; tindex++)
            {
               System.out.print(df.format(finalOutputs[tindex][eoindex]) + "    ");
               if ((tindex + 1) % numActs[outputLocation] == 0)
               {
                  System.out.println();

               }

            }

            if ((eoindex + 1) % numActs[outputLocation] == 0)
            {
               System.out.println();

            }
         }
      } // if (training)
      else
      {
         /**System.out.println("\nTable:        Input      Output");

         for (int k = 0; k < numTestCases; k++)
         {
            System.out.print("Test Case " + (k + 1) + ": |");

            for (int j = 0; j < numActs[INPUTS_LOC]; j++)
            {
               System.out.print(" " + String.format("%.0f", inputDataset[k][j]) + " ");
            }
            System.out.print("| - |");

            for (int j = 0; j < numActs[outputLocation]; j++)
            {
               System.out.print(" " + String.format("%.17f", finalOutputs[k][j]) + " ");
            }
            System.out.println("|");
         } // for (int k = 0; k < numTestCases; k++)
          */
         System.out.println("\n" + "----------------------------------------------" + "\n");
         System.out.println("Outputting Running Information...");

         System.out.println("\nExpected Values Against Actual Values: \n");

         for (int eoindex = 0; eoindex < numActs[outputLocation]; eoindex++)
         {
            for (int rindex = 0; rindex < numTestCases; rindex++)
            {
               System.out.print(df.format(finalOutputs[rindex][eoindex]) + "    ");
               if ((rindex + 1) % numActs[outputLocation] == 0)
               {
                  System.out.println();
               }
            }

            if ((eoindex + 1) % numActs[outputLocation] == 0)
            {
               System.out.println();
            }

         }
      } // else
   } //public void reportResults()

   /**
    * Main method.
    * Creates a network, sets parameters, allocates, populates, prints params, trains if intended to, and then runs and
    * prints results.
    */
   public static void main(String[] args)
   {
      Network network = new Network();

      String config;
      if (args.length > 0)
      {
         config = args[0];
      }
      else
      {
         config = DEFAULT_CONFIG;
         System.out.println("You did not pass any arguments. Will now use default configuration file");
      }

      network.setConfigParams(config);
      network.echoConfigParams();
      network.allocateArrays();
      network.populateArrays();

      if (network.training)
      {
         network.trainNetwork();
      }

      network.runNetwork();
      network.reportResults();

      if (network.saving)
      {
         network.saveWeights();
      }
   } //public static void main(String[] args)
} //public class Network

