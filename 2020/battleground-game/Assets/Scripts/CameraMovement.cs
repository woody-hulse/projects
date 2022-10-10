using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraMovement : MonoBehaviour
{

    public Transform playerTransform;
    public float speed;

    void Update()
    {
        Vector3 positionDiff = playerTransform.position - transform.position;

        if (positionDiff.magnitude > 0.1)
            transform.position += new Vector3(positionDiff.x, 0, positionDiff.z) / speed;
    }
}
