using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

[RequireComponent(typeof(NavMeshAgent))]

public class Enemy : LivingEntity
{
    public enum State {Idle, Chasing, Attacking};
    State currentState;

    Material skinMaterial;
    Color originalColor;

    LivingEntity targetEntity;
    bool hasTarget;

    NavMeshAgent pathfinder;
    Transform target;

    float attackDistanceThreshold = 1;
    float timeBetweenAttacks = 1;
    float nextAttackTime;

    float targetCollisionRadius;
    float enemyCollisionRadius;

    float damage = 1;

    Vector3 PDirection;

    protected override void Start()
    {
        base.Start();
        pathfinder = GetComponent<NavMeshAgent>();
        skinMaterial = GetComponent<Renderer>().material;
        originalColor = skinMaterial.color;

        if (GameObject.FindGameObjectWithTag("Player") != null)
        {
            currentState = State.Chasing;
            hasTarget = true;

            target = GameObject.FindGameObjectWithTag("Player").transform;
            targetEntity = target.GetComponent<LivingEntity>();
            targetEntity.OnDeath += OnTargetDeath;

            PDirection = Vector3.zero;

            StartCoroutine(UpdatePath());
        }

        enemyCollisionRadius = GetComponent<BoxCollider>().size.x / 2f;
        targetCollisionRadius = target.GetComponent<BoxCollider>().size.x / 2f;

    }

    void OnTargetDeath ()
    {
        hasTarget = false;
        currentState = State.Idle;
    }

    void Update()
    {
        if (hasTarget)
        {
            if (Time.time > nextAttackTime)
            {
                float squareDistanceToTarget = (target.position - transform.position).sqrMagnitude;

                if (squareDistanceToTarget < Mathf.Pow(attackDistanceThreshold + enemyCollisionRadius + targetCollisionRadius, 2))
                {
                    nextAttackTime = Time.time + timeBetweenAttacks;
                    StartCoroutine(Attack());
                }
            }
        }
    }

    IEnumerator Attack()
    {
        currentState = State.Attacking;
        pathfinder.enabled = false;

        Vector3 originalPosition = transform.position;
        Vector3 directionToTarget = (target.position - transform.position).normalized;
        Vector3 attackPosition = target.position - directionToTarget * enemyCollisionRadius;

        float percent = 0;
        float attackSpeed = 3;

        bool hasAppliedDamage = false;

        skinMaterial.color = Color.red;

        while (percent <= 1)
        {
            if (percent >= 0.5f && hasAppliedDamage == false)
            {
                hasAppliedDamage = true;
                targetEntity.TakeDamage(damage);
            }

            percent += Time.deltaTime * attackSpeed;
            float interpolation = (-Mathf.Pow(percent, 2) + percent) * 4;
            transform.position = Vector3.Lerp(originalPosition, attackPosition, interpolation);

            yield return null;
        }

        skinMaterial.color = originalColor;
        currentState = State.Chasing;
        pathfinder.enabled = true;
    }

    IEnumerator UpdatePath()
    {
        float refreshRate = 0.2f;

        while (hasTarget)
        {
            if (currentState == State.Chasing)
            {
                Vector3 directionToTarget = (target.position - transform.position).normalized;
                Vector3 targetPosition = target.position - directionToTarget * (targetCollisionRadius + enemyCollisionRadius);
                if (!dead)
                    pathfinder.SetDestination(targetPosition);

                Vector3 direction = targetPosition - transform.position;
                float rotationSpeed = 120;
                transform.eulerAngles += new Vector3(direction.x - PDirection.x, 0, direction.z - PDirection.z) * Time.deltaTime * rotationSpeed;
                PDirection = direction;
            }

            yield return new WaitForSeconds(refreshRate);
        }
    }
}
