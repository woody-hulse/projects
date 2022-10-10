using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class GameDifficulty
{
    static float secondsToMaxDifficulty = 80;

    public static float getDifficultyPercent()
    {
        return Mathf.Clamp01(Time.time / secondsToMaxDifficulty);
    }
}
