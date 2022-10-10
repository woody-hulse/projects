using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MapGenerator : MonoBehaviour
{
    public Map[] maps;
    public int mapIndex;

    public Transform tilePrefab;
    public Transform obstaclePrefab;
    public Vector2 maxMapSize;
    public Transform navmeshFloor;
    public Transform navmeshMaskPrefab;

    Transform[,] tileMap;

    public float tileSize;

    [Range(0, 1)]
    public float outlinePercent;

    List<Coordinate> allTileCoordinates;
    Queue<Coordinate> shuffledTileCoordinates;
    Queue<Coordinate> shuffledOpenCoordinates;

    Map currentMap;

    void Start()
    {
        FindObjectOfType<Spawner>().OnNewWave += OnNewWave;

    }

    void OnNewWave(int waveNumber)
    {
        mapIndex = waveNumber - 1;
        GenerateMap();
    }

    public void GenerateMap()
    {
        currentMap = maps[mapIndex];
        GetComponent<BoxCollider>().size = new Vector3(currentMap.mapSize.x * tileSize, 0.05f, currentMap.mapSize.y * tileSize);
        tileMap = new Transform[(int)currentMap.mapSize.x, (int)currentMap.mapSize.y];

        System.Random pseudoRandom = new System.Random(currentMap.seed);

        allTileCoordinates = new List<Coordinate>();

            //create map holder object
        string holderName = "Generated Map";
        if (transform.Find(holderName))
        {
            DestroyImmediate(transform.Find(holderName).gameObject);
        }
        Transform mapHolder = new GameObject(holderName).transform;
        mapHolder.parent = transform;

            //generate tiles
        for (int x = 0; x < currentMap.mapSize.x; x++)
        {
            for (int y = 0; y < currentMap.mapSize.y; y++)
            {
                allTileCoordinates.Add(new Coordinate(x, y));

                Vector3 tilePosition = CoordinateToPosition(x, y);
                Transform newTile = Instantiate(tilePrefab, tilePosition, Quaternion.Euler(Vector3.right * 90)) as Transform;
                newTile.localScale = Vector3.one * (1 - outlinePercent) * tileSize;
                newTile.parent = mapHolder;
                tileMap[x, y] = newTile;
            }
        }
        shuffledTileCoordinates = new Queue<Coordinate>(Utility.ShuffleArray<Coordinate>(allTileCoordinates.ToArray(), currentMap.seed));

            //spawn obstacles
        int obstacleCount = (int)(currentMap.mapSize.x * currentMap.mapSize.y * currentMap.obstaclePercent);
        int currentObstacleCount = 0;
        List<Coordinate> allOpenCoordinates = new List<Coordinate>(allTileCoordinates);

        bool[,] obstacleMap = new bool[(int)currentMap.mapSize.x, (int)currentMap.mapSize.y];

        for (int i = 0; i < obstacleCount; i++)
        {
            Coordinate randomCoordinate = GetRandomCoordinate();
            obstacleMap[randomCoordinate.x, randomCoordinate.y] = true;
            currentObstacleCount++;

            if (randomCoordinate != currentMap.center && MapIsAccessable(obstacleMap, currentObstacleCount))
            {
                float obstacleHeight = Mathf.Lerp((int)currentMap.minObstacleHeight, (int)currentMap.maxObstacleHeight, (float)pseudoRandom.NextDouble());
                Vector3 obstaclePosition = CoordinateToPosition(randomCoordinate.x, randomCoordinate.y);
                
                Transform newObstacle = Instantiate(obstaclePrefab, obstaclePosition + Vector3.up * obstacleHeight / 2f * (1 - outlinePercent), Quaternion.identity) as Transform;
                newObstacle.localScale = new Vector3((1 - outlinePercent) * tileSize, obstacleHeight, ((1 - outlinePercent) * tileSize));
                newObstacle.parent = mapHolder;

                allOpenCoordinates.Remove(randomCoordinate);
            }
            else
            {
                obstacleMap[randomCoordinate.x, randomCoordinate.y] = false;
                currentObstacleCount--;
            }
        }

        shuffledOpenCoordinates = new Queue<Coordinate>(Utility.ShuffleArray<Coordinate>(allOpenCoordinates.ToArray(), currentMap.seed));

        //draw navmesh boundaries
        Transform maskLeft = Instantiate(navmeshMaskPrefab, Vector3.left * (maxMapSize.x + currentMap.mapSize.x) / 4 * tileSize, Quaternion.identity) as Transform;
        maskLeft.parent = mapHolder;
        maskLeft.localScale = new Vector3((maxMapSize.x - currentMap.mapSize.x) / 2, 1, currentMap.mapSize.y) * tileSize;

        Transform maskRight = Instantiate(navmeshMaskPrefab, Vector3.right * (maxMapSize.x + currentMap.mapSize.x) / 4 * tileSize, Quaternion.identity) as Transform;
        maskRight.parent = mapHolder;
        maskRight.localScale = new Vector3((maxMapSize.x - currentMap.mapSize.x) / 2, 1, currentMap.mapSize.y) * tileSize;

        Transform maskUp = Instantiate(navmeshMaskPrefab, Vector3.forward * (maxMapSize.y + currentMap.mapSize.y) / 4 * tileSize, Quaternion.identity) as Transform;
        maskUp.parent = mapHolder;
        maskUp.localScale = new Vector3(maxMapSize.x, 1, (maxMapSize.y - currentMap.mapSize.y) / 2) * tileSize;

        Transform maskDown = Instantiate(navmeshMaskPrefab, Vector3.back * (maxMapSize.y + currentMap.mapSize.y) / 4 * tileSize, Quaternion.identity) as Transform;
        maskDown.parent = mapHolder;
        maskDown.localScale = new Vector3(maxMapSize.x, 1, (maxMapSize.y - currentMap.mapSize.y) / 2) * tileSize;

        navmeshFloor.localScale = new Vector3(maxMapSize.x, maxMapSize.y) * tileSize;
    }

    bool MapIsAccessable(bool[,] obstacleMap, int currentObstacleCount)
    {
        bool[,] seen = new bool[obstacleMap.GetLength(0), obstacleMap.GetLength(1)];
        Queue<Coordinate> queue = new Queue<Coordinate>();
        queue.Enqueue(currentMap.center);
        seen[currentMap.center.x, currentMap.center.y] = true;

        int accessibleTileCount = 1;

        while(queue.Count > 0)
        {
            Coordinate tile = queue.Dequeue();

            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    int adjX = tile.x + x;
                    int adjY = tile.y + y;

                    if (x == 0 || y == 0)
                    {
                        if (adjX >= 0 && adjX < obstacleMap.GetLength(0) && adjY >= 0 && adjY < obstacleMap.GetLength(1))
                        {
                            if (!seen[adjX, adjY] && !obstacleMap[adjX, adjY])
                            {
                                seen[adjX, adjY] = true;
                                queue.Enqueue(new Coordinate(adjX, adjY));
                                accessibleTileCount ++;
                            }
                        }
                    }
                }
            }
        }

        int targetAccessibleTileCount = (int)(currentMap.mapSize.x * currentMap.mapSize.y - currentObstacleCount);

        return targetAccessibleTileCount == accessibleTileCount;
    }

    Vector3 CoordinateToPosition(int x, int y)
    {
        return new Vector3(-currentMap.mapSize.x / 2 + x + 0.5f, 0, -currentMap.mapSize.y / 2 + y + 0.5f) * tileSize;
    }

    public Transform GetTileFromPosition(Vector3 pos)
    {
        int x = Mathf.RoundToInt(pos.x / tileSize + (currentMap.mapSize.x - 1) / 2f);
        int y = Mathf.RoundToInt(pos.z / tileSize + (currentMap.mapSize.y - 1) / 2f);

        x = Mathf.Clamp(x, 0, tileMap.GetLength(0));
        y = Mathf.Clamp(y, 0, tileMap.GetLength(1));

        return tileMap[x, y];
    }

    Coordinate GetRandomCoordinate()
    {
        Coordinate randomCoordinate = shuffledTileCoordinates.Dequeue();
        shuffledTileCoordinates.Enqueue(randomCoordinate);
        return randomCoordinate;
    }

    public Transform GetOpenTile()
    {
        Coordinate randomCoordinate = shuffledOpenCoordinates.Dequeue();
        shuffledOpenCoordinates.Enqueue(randomCoordinate);
        return tileMap[randomCoordinate.x, randomCoordinate.y];
    }

    public struct Coordinate
    {
        public int x;
        public int y;

        public Coordinate(int x, int y)
        {
            this.x = x;
            this.y = y;
        }

        public static bool operator ==(Coordinate c1, Coordinate c2)
        {
            return c1.x == c2.x && c2.y == c1.y;
        }

        public static bool operator !=(Coordinate c1, Coordinate c2)
        {
            return !(c1 == c2);
        }
    }

    [System.Serializable]
    public class Map
    {
        public Vector2 mapSize;
        [Range(0, 1)]
        public float obstaclePercent;
        public int seed;
        public float minObstacleHeight, maxObstacleHeight;
        public Color foreground, background;

        public Coordinate center
        {
            get
            {
                return new Coordinate((int)mapSize.x / 2, (int)mapSize.y / 2);
            }
        }
    }
}
