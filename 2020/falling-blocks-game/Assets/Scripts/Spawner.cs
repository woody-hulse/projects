using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Spawner : MonoBehaviour
{
    public GameObject fallingBlockPrefab;

    Vector2 screenHalfSizeInWorldUnits;

    public Vector2 spawnIntervalMinMax;
    float nextSpawnTime;

    public float spawnAngleMax;

    public Vector2 SpawnSizeMinMax;

    void Start()
    {

        screenHalfSizeInWorldUnits = new Vector2(Camera.main.aspect * Camera.main.orthographicSize, Camera.main.orthographicSize);
    }

    void Update()
    {

        if (Time.time > nextSpawnTime)
        {
            float spawnInterval = Mathf.Lerp(spawnIntervalMinMax.y, spawnIntervalMinMax.x, GameDifficulty.getDifficultyPercent());
            nextSpawnTime = Time.time + spawnInterval;

            float spawnAngle = Random.Range(-spawnAngleMax, spawnAngleMax);
            float spawnSize = Random.Range(SpawnSizeMinMax.x, SpawnSizeMinMax.y);
            Vector2 spawnPosition = new Vector2(Random.Range(-screenHalfSizeInWorldUnits.x, screenHalfSizeInWorldUnits.x), screenHalfSizeInWorldUnits.y + spawnSize);
            GameObject newBlock = (GameObject)Instantiate(fallingBlockPrefab, spawnPosition, Quaternion.Euler(Vector3.forward * spawnAngle));

            newBlock.transform.localScale = Vector2.one * spawnSize;
        }
    }
}
