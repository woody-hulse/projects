using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class PlayerController : MonoBehaviour
{
    Rigidbody rb;

    Vector3 velocity;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }


    public void Move(Vector3 velocity)
    {
        this.velocity = velocity;
    }

    public void LookAt(Vector3 point)
    {
        Vector3 raisedPoint = new Vector3(point.x, transform.position.y, point.z);
        transform.LookAt(raisedPoint);
    }

    void FixedUpdate()
    {
        rb.MovePosition(rb.position + velocity * Time.fixedDeltaTime);
    }
}
