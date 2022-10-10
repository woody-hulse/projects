using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    int speed = 8;

    public event System.Action OnPlayerDeath;

    public GameObject particlePrefab;

    float screenHalfWidthInWorldUnits;

    void Start()
    {
        transform.position = new Vector3(0, -5, 0);

        screenHalfWidthInWorldUnits = Camera.main.aspect * Camera.main.orthographicSize - 0.5f;
    }

    void Update()
    {
        float inputX = Input.GetAxisRaw("Horizontal");
        float velocity = inputX * speed;

        if ((Input.GetAxisRaw("Horizontal") > 0 && transform.position.x < screenHalfWidthInWorldUnits) ||
            (Input.GetAxisRaw("Horizontal") < 0 && transform.position.x > -screenHalfWidthInWorldUnits))
            transform.Translate(Vector2.right * velocity * Time.deltaTime);
    }

    private void OnTriggerEnter2D(Collider2D triggerCollider)
    {
        if (triggerCollider.tag == "FallingBlock")
        {
            if (OnPlayerDeath != null)
                OnPlayerDeath();

            for (int i = 0; i < 6; i++)
            {
                float spawnAngle = Random.Range(-180, 180);
                float spawnSize = Random.Range(0.2f, 0.9f);
                Vector2 spawnPosition = transform.position;
                GameObject newBlock = (GameObject)Instantiate(particlePrefab, spawnPosition, Quaternion.Euler(Vector3.forward * spawnAngle));

                newBlock.transform.localScale = Vector2.one * spawnSize;
            }
            Destroy(gameObject);
        }
    }
}
