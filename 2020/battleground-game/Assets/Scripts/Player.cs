using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(PlayerController))]
[RequireComponent(typeof(GunController))]

public class Player : LivingEntity
{

    public float MovementSpeed = 5;

    Camera viewCamera;
    PlayerController controller;
    GunController gunController;

    float rotation;
    float rotationSpeed = 2;

    protected override void Start()
    {
        base.Start();
        controller = GetComponent<PlayerController>();
        gunController = GetComponent<GunController>();
        viewCamera = Camera.main;
    }

   
    void Update()
    {
        //movement input
        Vector3 moveInput = new Vector3(Input.GetAxisRaw("Horizontal"), 0, Input.GetAxisRaw("Vertical"));
        Vector3 velocity = moveInput.normalized * MovementSpeed;
        controller.Move(velocity);

        transform.Rotate(0, rotation * rotationSpeed, 0);

            //look input
        Ray ray = viewCamera.ScreenPointToRay(Input.mousePosition);
        Plane groundPlane = new Plane(Vector3.up, Vector3.zero);
        float rayDistance;

        if (groundPlane.Raycast(ray, out rayDistance))
        {
            Vector3 point = ray.GetPoint(rayDistance);
            controller.LookAt(point);
        }
        //weapon input;
        if (Input.GetMouseButton(0))
            gunController.Shoot();

        if (transform.position.y < -10)
            fallDamage();
    }
}
