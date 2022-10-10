using System.Collections;
using System.Collections.Generic;

public static class Utility
{
    public static T[] ShuffleArray<T>(T[] array, int seed)
    {
        System.Random random = new System.Random(seed);

        for (int i = 0; i < array.Length - 1; i++)
        {
            int randomIndex = random.Next(i, array.Length);
            T temporaryItem = array[i];
            array[i] = array[randomIndex];
            array[randomIndex] = temporaryItem;
        }

        return array;
    }
}
