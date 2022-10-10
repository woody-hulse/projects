using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Particle : MonoBehaviour
{

    Vector2 velocity;
    float speed;

    float rotation;
    float forward;

    void Start()
    {
        velocity = new Vector2(Random.Range(-6, 6), Random.Range(-6, 6));

        forward = Random.Range(-1, 1);
    }

    void Update()
    {

        velocity *= 0.95f;

        transform.Translate(velocity * Time.deltaTime);

        speed = velocity.magnitude;
        transform.localScale = new Vector3(speed / 5, speed / 5, speed / 5);
        rotation += speed/2;

        transform.Rotate(0, 0, rotation*forward);

        if (speed <= 0.1f)
            Destroy(gameObject);
    }
}
