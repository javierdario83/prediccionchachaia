# APIs externas para lesiones, alineaciones y xG

Estas fuentes no sustituyen a `football-data.co.uk` en el MVP: sirven como integración futura para mejorar el contexto del modelo con datos que el CSV público no trae de forma completa.

## Resumen recomendado

| Proveedor | Lesiones / suspendidos | Alineaciones | xG | Tipo | Recomendación |
| --- | --- | --- | --- | --- | --- |
| Sportmonks | Sí, vía `sidelined.sideline` | Sí, vía `lineups`, y expected lineups como add-on | Sí, endpoints Expected/xG y xG por jugador en lineups | Pago / trial | Mejor opción si el cliente acepta pagar por datos premium. |
| API-Football / API-Sports | Sí, endpoint injuries | Sí, lineups y fixture players | Inconsistente; verificar liga/plan/endpoints | Freemium | Buena segunda opción por costo/cobertura, pero validar xG antes. |
| TheStatsAPI | No confirmado en esta revisión | Match/player stats | Sí, xG y stats históricos | Trial/pago | Buena opción para xG histórico y modelos, menos claro para lesiones. |
| foot.io | No confirmado | Sí, lineups en base pública | Sí, eventos shot-level xG | Free tier / API key opcional | Interesante para investigación o prototipo xG si la cobertura encaja. |
| football-data.org | No profundo | Squads/lineups/subs según cobertura | No enfocado a xG | Free + planes | Útil como API JSON auxiliar, no como fuente principal de xG. |

## Sportmonks

Documentación revisada:

- Lineups y formaciones: `https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/lineups-and-formations`
- Expected lineups: `https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/premium-expected-lineups/get-expected-lineup-by-team`
- Índice de endpoints: `https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints`

Puntos importantes:

- Permite pedir `formations`, `lineups` y `sidelined.sideline` para lesionados/suspendidos.
- La documentación conecta lineups con xG por jugador mediante `lineups.xGLineup`.
- Tiene endpoints de Expected/xG y expected lineups premium.
- Es la fuente más completa de las revisadas para lesiones + alineaciones + xG, pero hay que revisar plan/precio.

## API-Football / API-Sports

Documentación/recursos revisados:

- Sitio principal: `https://www.api-football.com/`
- Documentación: `https://www.api-football.com/documentation-v3`
- Guía API-Football: `https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide`

Puntos importantes:

- Expone endpoints de injuries, lineups, fixture statistics, player statistics, predictions y odds.
- El endpoint injuries permite filtrar por fixture, team, player, league/season o date según la guía.
- Para xG hay que validar muy bien cobertura y plan; no conviene asumir que todas las ligas tienen xG consistente.

## TheStatsAPI

Documentación revisada:

- Sitio principal: `https://www.thestatsapi.com/`

Puntos importantes:

- Ofrece match stats, player stats, odds, xG, team stats e histórico de 10 años según su página.
- Parece más fuerte para xG y stats históricos que para lesiones/alineaciones probables.
- Requiere trial/API key y validar endpoints exactos antes de integrarlo.

## foot.io

Documentación revisada:

- Docs: `https://foot.io/docs`

Puntos importantes:

- Public reads funcionan sin key con rate limit por IP; el uso autenticado utiliza header `x-api-key`.
- Indica base grande con matches, players, lineups y eventos shot-level xG.
- Puede servir para prototipo o investigación si la cobertura de ligas coincide con el producto.

## football-data.org

Documentación/sitio revisado:

- Sitio principal: `https://www.football-data.org/`

Puntos importantes:

- API REST gratuita para competiciones principales, live scores, fixtures, tablas, squads y lineups/subs.
- No es la mejor fuente para xG, lesiones o alineaciones probables profundas.
- Puede complementar fixtures/teams en JSON, pero no reemplaza Sportmonks/API-Football para datos premium.

## Integración propuesta en el código

1. Mantener `football-data.co.uk` como fuente base gratuita.
2. Crear adaptadores opcionales:
   - `football_predictor/providers/sportmonks.py`
   - `football_predictor/providers/api_football.py`
   - `football_predictor/providers/footio.py`
3. Guardar datos externos normalizados en tablas:
   - `external_injuries`
   - `external_lineups`
   - `external_xg`
4. Agregar esas señales al score:
   - bajar confianza si faltan titulares clave,
   - subir/bajar expected goals según xG reciente,
   - evitar picks si las alineaciones son inciertas,
   - mejorar BTTS/Over con xG a favor/en contra.

## Orden sugerido

1. Primero integrar Sportmonks si el cliente pagará datos premium.
2. Si se busca freemium, probar API-Football para lesiones/alineaciones y validar xG por liga.
3. Si se busca xG histórico barato/gratuito, evaluar foot.io o TheStatsAPI según cobertura.
