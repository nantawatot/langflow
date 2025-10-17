# Golf Taxonomy

# Golf Sports Taxonomy
```json
{
    "level 1": {
        "category": "Golf Sports",
        "description": "A precision ball sport where players use clubs to hit a ball into a series of holes on a course in as few strokes as possible."
    },
    "level 2": {
        "category": "Golf Stroke Play",
        "description": "Competition format where players aim to complete the course with the fewest total strokes.",
        "reference_website": "https://www.topendsports.com/sport/list/golf.htm"
    }
}
```
### Core Attributes:
- **Sport Category**: Golf Sports, The general category of the sport that includes all types of golf competitions and formats.
- **Sport**: Golf, A precision ball sport where players use clubs to hit a ball into a series of holes on a course in as few strokes as possible. It is played professionally in various individual and team formats across multiple tournaments.
- **Discipline Type**: Golf Stroke Play, The most common scoring format in golf. Players compete individually to complete a course using the fewest total strokes over one or more rounds.
- **Competition/Event**: Official tournament or event name where stroke play is used (e.g., The Open Championship, The Masters, U.S. Open). These are often part of professional tours such as PGA or LPGA.
- **Season/Year**: The year or season in which the competition is held. This is useful for organizing data chronologically, e.g., The Open Championship 2023.
- **Round/Stage**: Specific phases or segments of a tournament:
  - Round 1, Round 2 – Early stages of the event.
  - Final Round – The last round, determining the final standings.
- **Match**: Player face-offs, e.g., Tiger Woods vs Rory McIlroy
- **Player/Team**: A professional golfer or pair/team (for formats like foursomes or four-ball). Examples include Tiger Woods, Rory McIlroy.
  - Role/Position → Player’s role, e.g., Golfer.
  - Nationality → Player’s country, e.g., USA, Northern Ireland.
  - Coach → Non-playing personnel who are crucial to the player’s performance, such as coaches, managers, or trainers.
- **Management**: Management of the event, including:
  - **Governing Body**: The organization responsible for overseeing the rules and regulations of the sport and the event (e.g., PGA Tour, LPGA).
- **Broadcast**: The media outlets responsible for televising or streaming the event (e.g., ESPN, NBC Sports).
- **Sponsors**: Brands or corporations that provide financial support to players or events (e.g., Rolex, Nike, Titleist, Mastercard).
- **Venue**: The golf course and location where the event is held:
  - Location – Geographic place or club name (e.g., St Andrews, Augusta National).
  - Course Type – Design or environmental style of the course (e.g., Links Course, Parkland Course, Desert Course).



### Example Taxonomy Structure for Golf:
```plaintext
Golf → Sport Category
A precision ball sport where players use clubs to hit a ball into a series of holes on a course in as few strokes as possible. It is played professionally in various individual and team formats across multiple tournaments.

└── Golf Stroke Play → The most common scoring format in golf. Players compete individually to complete a course using the fewest total strokes over one or more rounds.

    └── Competition/Event → Official tournament or event name using stroke play.
        ├── Examples → The Masters, The Open Championship, U.S. Open, PGA Championship.
        ├── Tour Affiliation → Associated professional tours (e.g., PGA Tour, LPGA Tour).

        └── Round/Stage → Specific phases or segments of the tournament.
            ├── Round 1 → Opening round of the competition.
            ├── Round 2 → Second round, may include cut line enforcement.
            ├── Round 3 → Often referred to as "Moving Day."
            └── Final Round → Last round, determining final standings.

            └── Venue → The golf course and its location.
                ├── Location → Name and geography (e.g., Augusta National Golf Club, Georgia, USA).
                └── Course Type → Style/design of the course:
                    ├── Links Course → Coastal, sandy terrain with few trees (e.g., St Andrews).
                    ├── Parkland Course → Inland, lush, tree-lined (e.g., Augusta National).
                    └── Desert Course → Arid terrain with rocky features (e.g., TPC Scottsdale).

        └── Player/Team → Individual golfer or paired group depending on format.
            ├── Player Name → Example: Tiger Woods, Nelly Korda.
            ├── Role/Position → Always "Golfer" in this format.
            ├── Stats → Tournament stats like stroke total, fairways hit, greens in regulation.
            └── Sponsors → Personal or event sponsors (e.g., Nike, Titleist, Rolex, Mastercard).

        └── Management → Organizations responsible for operations and governance.
            ├── Governing Body → Rules and oversight body (e.g., USGA, R&A, PGA Tour).
            ├── Tournament Director → In-charge of event logistics and player services.
            └── Referee/Officials → Enforce rules and handle disputes on course.

        └── Broadcast → Media outlets covering the event.
            ├── TV Networks → ESPN, NBC Sports, Sky Sports, Golf Channel.
            └── Streaming Platforms → ESPN+, Peacock, Masters.com.

        └── Sponsors → Event-level sponsors providing financial or commercial backing.
            ├── Title Sponsor → Brand in event name (e.g., RBC Canadian Open).
            └── Secondary Sponsors → Rolex, FedEx, AT&T, Mastercard.

        └── Season/Year → When the tournament was held.
            ├── Example → The Open Championship 2023.
            └── Purpose → Helps organize historical and seasonal data.

```