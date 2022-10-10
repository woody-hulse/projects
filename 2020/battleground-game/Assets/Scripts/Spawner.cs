using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Spawner : MonoBehaviour
{
    public Wave[] waves;
    public Enemy enemy;

    LivingEntity playerEntity;
    Transform playerTransform;

    Wave currentWave;
    int currentWaveNumber;

    int enemiesRemainingToSpawn;
    int enemiesRemainingAlive;
    float nextSpawnTime;

    float timeBetweenPlayerCheck = 2;
    float nextPlayerCheck;
    float playerThresholdDistance = 1.5f;
    Vector3 pcampPosition;
    bool isCamping;

    bool isDisabled;

    public event System.Action<int> OnNewWave;

    MapGenerator map;

    void Start()
    {
        playerEntity = FindObjectOfType<Player>();
        playerEntity.OnDeath += OnPlayerDeath;
        playerTransform = playerEntity.transform;

        nextPlayerCheck = timeBetweenPlayerCheck + Time.time;
        pcampPosition = playerTransform.position;

        map = FindObjectOfType<MapGenerator>();

        NextWave();
    }

    void Update()
    {
        if (!isDisabled)
        {
            if (Time.time > nextPlayerCheck)
            {
                nextPlayerCheck = Time.time + timeBetweenPlayerCheck;

                isCamping = (Vector3.Distance(pcampPosition, playerTransform.position) < playerThresholdDistance);
                pcampPosition = playerTransform.position;
            }

            if (enemiesRemainingToSpawn > 0 && Time.time > nextSpawnTime)
            {
                enemiesRemainingToSpawn--;
                nextSpawnTime = Time.time + currentWave.timeBetweenSpawns;

                StartCoroutine(SpawnEnemy());
            }
        }
    }

    IEnumerator SpawnEnemy()
    {
        float spawnDelay = 1;
        float tileFlashSpeed = 4;

        Transform spawnTile = map.GetOpenTile();

        if (isCamping)
        {
            spawnTile = map.GetTileFromPosition(playerTransform.position);
        }

        Material spawnTileMaterial = spawnTile.GetComponent<Renderer>().material;
        Color initialColor = spawnTileMaterial.color;
        Color flashColor = Color.red;
        float spawnTimer = 0;

        while (spawnTimer < spawnDelay)
        {
            spawnTileMaterial.color = Color.Lerp(initialColor, flashColor, Mathf.PingPong(spawnTimer * tileFlashSpeed, 1));

            spawnTimer += Time.deltaTime;
            yield return null;
        }

        Enemy spawnedEnemy = Instantiate(enemy, spawnTile.position + Vector3.up * 0.5f, Quaternion.identity) as Enemy;
        spawnedEnemy.OnDeath += OnEnemyDeath;
    }

    void ResetPlayerPosition()
    {
        //playerTransform.position = map.GetTileFromPosition(Vector3.zero).position + Vector3.up * 3;
        playerTransform.position = Vector3.zero + Vector3.up * 0.5f;
    }

    void OnPlayerDeath()
    {
        isDisabled = true;
    }

    void OnEnemyDeath()
    {
        enemiesRemainingAlive--;

        if (enemiesRemainingAlive == 0)
        {
            NextWave();
        }
    }

    void NextWave()
    {
        ResetPlayerPosition();

        currentWaveNumber++;
        if (currentWaveNumber - 1 < waves.Length)
        {
            currentWave = waves[currentWaveNumber - 1];

            enemiesRemainingToSpawn = currentWave.enemyCount;
            enemiesRemainingAlive = enemiesRemainingToSpawn;
        }

        if (OnNewWave != null)
            OnNewWave(currentWaveNumber);
    }

    [System.Serializable]
    public class Wave
    {
        public int enemyCount;
        public float timeBetweenSpawns;
    }
}
