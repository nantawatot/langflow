# Ice Hockey Taxonomy
```json
{
    "level 1": {
        "category": "Hockey Sports",
        "description": "such as field hockey, ice hockey and roller hockey."
    },
    "level 2": {
        "category": "Ice Hockey",
        "description": "A fast-paced sport played on an ice rink with six players per team, aiming to score by shooting a puck into the opponent\u2019s goal.",
        "reference_website": "https://www.topendsports.com/sport/list/hockey-ice.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Hockey Sports, The general category of the sport that includes all types of hockey competitions and formats.
- **Sport**: Ice Hockey, A fast-paced sport played on an ice rink with six players per team, aiming to score by shooting a puck into the opponent’s goal.
- **Focused Subcategory**: Professional Ice Hockey, The specific variant of ice hockey played at the professional level, typically in leagues like the NHL.
- **Discipline Type**: Competition formats, e.g., Regular Season, Playoffs, All-Star Game, Exhibition Match.
- **Competition/Event**: Tournament names, e.g., NHL Stanley Cup Playoffs, Winter Classic, IIHF World Championship.
- **Season/Year**: Year the event occurs, e.g., 2023, 2024.
- **Stage/Match/Round**: Competition stages, e.g., Round 1, Semi-final, Final.
- **Team/Club**: Participating teams, e.g., Toronto Maple Leafs, Boston Bruins, Montreal Canadiens, New York Rangers.
  - **Athlete**: Player names, e.g., Sidney Crosby, Alexander Ovechkin, Connor McDavid, Patrick Kane.
    - **Role/Position**: Player roles, e.g., Forward, Defenseman, Goaltender, Captain, Alternate Captain.
    - **Equipment/Brand/Model**: Player gear, e.g., CCM Hockey Stick, Bauer Skates, Warrior Gloves.
  - **Management**: Team officials, e.g., Head Coach, General Manager, Assistant Coach.
  - **Sponsors**: Team sponsors, e.g., Adidas, Coca-Cola, Pepsi, Honda, KIA.
- **Arena/Venue**: Event locations, e.g., Madison Square Garden, Rogers Arena, Bell Centre, TD Garden.
- **League**: Leagues, e.g., NHL (National Hockey League), KHL (Kontinental Hockey League), SHL (Swedish Hockey League).
- **Referee/Official**: Match officials, e.g., Head Referee, Linesman, Officiating Crew.


### Example Taxonomy Structure for Ice Hockey:
```plaintext
Hockey Sports → Sport Category
Hockey is a high-intensity team sport played on ice or field. This taxonomy focuses on **Ice Hockey**, known for its speed, physicality, and strategic gameplay, especially popular in North America and Europe.

└── Ice Hockey → The specific discipline covered here, played on ice using sticks and a puck.
    └── Discipline Type → Types of organized competitive formats.
        ├── Regular Season → Standard league games for ranking and playoff seeding.
        ├── Playoffs → Elimination rounds leading to a championship title.
        ├── All-Star Game → Exhibition match featuring star players.
        └── Exhibition Match → Preseason or friendly games.

        └── Competition/Event → Prominent tournaments and league events.
            ├── Name → e.g., NHL Stanley Cup Playoffs, Winter Classic, IIHF World Championship.

            └── League → Governing bodies or leagues managing the competition.
                ├── NHL → National Hockey League (North America)
                ├── KHL → Kontinental Hockey League (Europe/Asia)
                └── SHL → Swedish Hockey League (Sweden)

            ├── Season/Year → The competitive timeframe, e.g., 2023–2024 season.

            └── Arena/Venue → Stadiums and rinks where games are played.
                ├── Name → e.g., Madison Square Garden, Rogers Arena, Bell Centre, TD Garden.
                └── Location → City and country.

            └── Stage/Match/Round → Phases or individual games.
                ├── Examples → Round 1, Quarterfinals, Semifinals, Final.

            └── Team/Club → Teams participating in events.
                ├── Name → e.g., Toronto Maple Leafs, Boston Bruins, Montreal Canadiens, New York Rangers.

                └── Athlete → Individual players on the team.
                    ├── Name → e.g., Sidney Crosby, Alexander Ovechkin, Connor McDavid, Patrick Kane.

                    └── Role/Position →
                        ├── Forward
                        ├── Defenseman
                        ├── Goaltender
                        ├── Captain
                        └── Alternate Captain

                    └── Equipment/Brand/Model →
                        ├── Sticks → e.g., CCM Ribcor, Bauer Vapor
                        ├── Skates → e.g., Bauer Supreme, CCM Tacks
                        └── Gloves → e.g., Warrior Alpha, CCM Jetspeed

                    └── Player Stats →
                        ├── Goals
                        ├── Assists
                        ├── Points
                        ├── Shots on Goal
                        ├── Plus-Minus
                        └── Save Percentage (for goalies)

                └── Management →
                    ├── Head Coach
                    ├── Assistant Coach
                    ├── General Manager
                    └── Team Doctor / Trainer

                └── Sponsors →
                    ├── Global Sponsors → e.g., Adidas, Pepsi, Honda
                    └── Team Sponsors → e.g., KIA, Scotiabank

            └── Referee/Official →
                ├── Head Referee
                ├── Linesman
                └── Officiating Crew


```