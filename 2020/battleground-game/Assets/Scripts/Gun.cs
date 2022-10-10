using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Gun : MonoBehaviour
{
    public Transform barrel;
    public Projectile projectile;
    public float msBetweenShots = 100;
    public float barrelVelocity = 35;

    float nextShotTime;

    public void Shoot()
    {
        if (Time.time > nextShotTime)
        {
            nextShotTime = Time.time + msBetweenShots / 1000f;
            Projectile newProjectile = Instantiate(projectile, barrel.position, barrel.rotation) as Projectile;
            newProjectile.SetSpeed(barrelVelocity);
        }
    }
}
