using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class UIManager : MonoBehaviour
{
    public GameObject GameOver;
    public GameObject GameWin;

    bool gameIsOver;

    void Start()
    {
        Guard.SeenPlayer += ShowGameOver;
        Player.reachedEnd += ShowGameWin;
    }

    void Update()
    {
        if(gameIsOver)
        {
            if (Input.GetKeyDown(KeyCode.Space))
            {
                SceneManager.LoadScene(0);
            }
        }
    }

    void ShowGameWin()
    {
        OnGameOver(GameWin);
    }

    void ShowGameOver()
    {
        OnGameOver(GameOver);
    }

    void OnGameOver(GameObject gameUI)
    {
        gameUI.SetActive(true);
        gameIsOver = true;
        Guard.SeenPlayer -= ShowGameOver;
        Player.reachedEnd -= ShowGameWin;
    }
}
