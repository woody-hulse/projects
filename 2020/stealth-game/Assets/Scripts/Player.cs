using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Player : MonoBehaviour
{
    public static event System.Action reachedEnd;

    public float movementSpeed = 7;

    public float smoothMoveTime = 0.1f;
    public float turnSpeed = 8;

    float angle;
    float smoothInputMagnitude;
    float smoothMoveVelocity;
    Vector3 velocity;

    Rigidbody rb;

    bool disabled;

    public Transform goal;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        Guard.SeenPlayer += Disabled;
        reachedEnd += Disabled;
    }


    void Update()
    {
        Vector3 inputDirection = Vector3.zero;

        if (!disabled)
        {
            inputDirection = new Vector3(Input.GetAxisRaw("Horizontal"), 0, Input.GetAxisRaw("Vertical")).normalized;
        }

        float inputMagnitude = inputDirection.magnitude;
        smoothInputMagnitude = Mathf.SmoothDamp(smoothInputMagnitude, inputMagnitude, ref smoothMoveVelocity, smoothMoveTime);

        float targetAngle = Mathf.Atan2(inputDirection.x, inputDirection.z) * Mathf.Rad2Deg;
        angle = Mathf.LerpAngle(angle, targetAngle, turnSpeed * Time.deltaTime * inputMagnitude);
        transform.eulerAngles = Vector3.up * angle;

        transform.Translate(transform.forward * movementSpeed * Time.deltaTime * smoothInputMagnitude, Space.World);

        velocity = transform.forward * movementSpeed * smoothInputMagnitude;

        if (Vector3.Distance(transform.position, goal.position) < transform.localScale.x)
            if (reachedEnd != null)
                reachedEnd();
    }

    void Disabled()
    {
        disabled = true;
    }

    private void FixedUpdate()
    {
        rb.MoveRotation(Quaternion.Euler(Vector3.up * angle));
        rb.MovePosition(rb.position + velocity * Time.deltaTime);
    }

    void OnDestroy()
    {
        Guard.SeenPlayer -= Disabled;
        reachedEnd -= Disabled;
    }
}
