# Plan d'Implémentation de Résolution du Bug de Déconnexion en Visio

> **Pour l'Agent :** Implémentez ce plan tâche par tâche avec validation entre les étapes.

**Objectif :** Résoudre le bug de déconnexion automatique après ~1 minute de visio provoqué par l'arrêt silencieux de la boucle de rendu suite à une exception WebGL (starvation de trames WebRTC).

**Architecture :** Sécuriser la boucle chaude de rendu dans `AdvancedMattingProcessor.ts` en entourant l'appel `_renderFrame()` d'un bloc `try/catch`. Cela garantit que toute exception WebGL temporaire (par ex. perte de contexte ou erreur de compilation temporaire) n'interrompt pas la planification des frames suivantes.

**Stack Technique :** TypeScript, WebGL2, requestVideoFrameCallback, LiveKit WebRTC.

---

### Tâche 1 : Sécurisation de la boucle de rendu (requestVideoFrameCallback & fallback)

**Fichiers :**
- Modifier : `src/frontend/src/features/rooms/livekit/components/blur/AdvancedMattingProcessor.ts:663-685`

**Étape 1 : Écrire l'implémentation sécurisée**

Dans la méthode `_scheduleRender()`, encadrer chaque appel à `this._renderFrame()` avec un bloc `try/catch` de sécurité.

```typescript
  private _scheduleRender(): void {
    if (!this._segLoopActive || this._destroyed) return
    
    // Modern synchronization: requestVideoFrameCallback
    if (this.videoElement && 'requestVideoFrameCallback' in this.videoElement) {
      this._renderLoopHandle = (this.videoElement as any).requestVideoFrameCallback(() => {
        if (!this._segLoopActive || this._destroyed) return
        try {
          this._renderFrame()
        } catch (e) {
          console.error('[AdvancedMattingProcessor] Render frame failed, skipping:', e)
        }
        this._scheduleRender()
      })
      return
    }

    // Fallback: requestAnimationFrame with 50fps capping
    this._renderLoopHandle = requestAnimationFrame((now) => {
      if (!this._segLoopActive || this._destroyed) return
      if (now - this._lastRenderTime >= AdvancedMattingProcessor.RENDER_TARGET_MS) {
        this._lastRenderTime = now
        try {
          this._renderFrame()
        } catch (e) {
          console.error('[AdvancedMattingProcessor] Render frame failed (fallback), skipping:', e)
        }
      }
      this._scheduleRender()
    })
  }
```

**Étape 2 : Commit**

```bash
git add src/frontend/src/features/rooms/livekit/components/blur/AdvancedMattingProcessor.ts
git commit -m "fix(blur): wrap render frame execution in try/catch to prevent loop starvation"
```
