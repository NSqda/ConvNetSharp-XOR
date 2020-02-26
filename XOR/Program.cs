using System;
using System.Collections.Generic;
using ConvNetSharp;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;

namespace XOR
{
    class Program
    {
        static private Func<double, int> isOne = (x) => x > 0.5 ? 1 : 0;
        static void Main(string[] args)
        {
            Program program = new Program();

            var net = program.XOR();

            var input = BuilderInstance.Volume.SameAs(new Shape(1, 1, 2));

            do
            {
                Console.WriteLine("Type two numbers to calculate XOR (ex. 0 0 )");
                Console.WriteLine("To escape test, type \'exit\' (ex. exit)");
                var line = Console.ReadLine().Split();

                if (line.Length == 2)
                {
                    for (var i = 0; i < 2; i++)
                        input.Set(0, 0, i, double.Parse(line[i]));

                    var result = net.Forward(input);

                    Console.WriteLine(isOne(result.Get(0)));
                }

                if (line[0].Equals("exit"))
                    break;
            } while (true);

        }



        public Net<double> XOR()
        {

            var network = new Net<double>();
            network.AddLayer(new InputLayer(1, 1, 2));
            network.AddLayer(new FullyConnLayer(6));
            network.AddLayer(new ReluLayer());
            network.AddLayer(new FullyConnLayer(2));
            network.AddLayer(new ReluLayer());
            network.AddLayer(new RegressionLayer());

            List<int[]> data = new List<int[]>();
            List<int> label = new List<int>();

            data.Add(new int[] { 0, 0 });
            label.Add(0);

            data.Add(new[] { 0, 1 });
            label.Add(1);

            data.Add(new[] { 1, 0 });
            label.Add(1);

            data.Add(new[] { 1, 1 });
            label.Add(0);

            var trainer = new SgdTrainer<double>(network) { LearningRate = 0.01, BatchSize = label.Count };

            var n = label.Count;

            var x = BuilderInstance.Volume.SameAs(new Shape(1, 1, 2, n));
            var y = BuilderInstance.Volume.SameAs(new Shape(1, 1, 2, n));

            for (var i = 0; i < n; i++)
            {
                y.Set(0, 0, label[i], i, 1.0);

                x.Set(0, 0, 0, i, data[i][0]);
                x.Set(0, 0, 1, i, data[i][1]);
            }

            do
            {
                var avloss = 0.0;

                trainer.Train(x, y);
                avloss = trainer.Loss;

                //avloss /= 50.0;
                Console.WriteLine(" Loss:" + avloss);
            } while (!Console.KeyAvailable);


            var input = BuilderInstance.Volume.SameAs(new Shape(1, 1, 2, n));
            for (var i = 0; i < n; i++)
                for (var i2 = 0; i2 < 2; i2++)
                    input.Set(0, 0, i2, i, data[i][i2]);

            var result = network.Forward(input);

            for (int i = 0; i < n; i++)
                Console.WriteLine("{0} XOR {1} = {2}", data[i][0], data[i][1], isOne(result.Get(i)));

            return network;
        }
    }
}
