# Lacrosse Taxonomy
```json
{
    "level 1": {
        "category": "Hockey Sports",
        "description": "such as field hockey, ice hockey and roller hockey."
    },
    "level 2": {
        "category": "Lacrosse",
        "description": "a team game, originally played by North American Indians, in which the ball is thrown, caught, and carried with a long-handled stick with a piece of shallow netting at one end. Versions include Field Lacrosse, Box Lacrosse, Women's Lacrosse, Sixes Lacrosse.",
        "reference_website": "https://www.topendsports.com/sport/list/lacrosse.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Lacrosse Sports, The general category of the sport that includes all types of lacrosse competitions and formats.
- **Sport**: Lacrosse, A team game, originally played by North American Indians, in which the ball is thrown, caught, and carried with a long-handled stick with a piece of shallow netting at one end. Versions include Field Lacrosse, Box Lacrosse.
- **Discipline Type**: Various competition styles including:
  - Field Lacrosse – Played outdoors on a large field with 10 players per team.
  - Box Lacrosse – Indoor version played in a smaller enclosed area with 6 players per team.- **Competition/Event**: Tournament names, e.g., NCAA Lacrosse Championship, World Lacrosse Championship, Major League Lacrosse.
- **Season/Year**: Year the event occurs, e.g., 2023.
- **Round/Match**: Competition stages, e.g., Quarterfinals, Semifinals, Final.
- **Team/Club**: Participating teams, e.g., Team USA, Canada, Philadelphia Wings.
  - **Athlete**: Player names, e.g., Lyle Thompson, Tom Schreiber, Trevor Baptiste.
    - **Role/Position**: Player roles, e.g., Attackman, Midfielder, Defenseman, Goalkeeper.
    - **Equipment/Brand/Model**: Player gear, e.g., Lacrosse Stick, Helmet such as Warrior Evo, STX Surgeon.
  - **Management**: Team officials, e.g., Head Coach, Assistant Coach, General Manager.
  - **Sponsors**: Team sponsors, e.g., Nike, Adidas, Under Armour.
- **Ranking**: Team rankings, e.g., World Ranking, NCAA Ranking.
- **Venue**: Event locations, e.g., Gillette Stadium, M&T Bank Stadium.
- **Prize Money/Rewards**: Prize amounts or awards, e.g., $10,000, Gold Medal.
- **Federation**: Organizing bodies, e.g., World Lacrosse, NCAA.
- **Club/League**: Associated leagues or clubs, e.g., National Lacrosse League (NLL), Premier Lacrosse League (PLL).
- **Tournament Type**: Competition types, e.g., Invitational, Championship, Exhibition.
- **Broadcasting**: Media outlets broadcasting games, e.g., ESPN, TSN, Sportsnet.


### Example Taxonomy Structure for Lacrosse:
```plaintext
Lacrosse → Sport Category
A fast-paced team sport combining speed, strategy, and physical contact. It is played with a stick (crosse) and a ball, with formats including outdoor field lacrosse and indoor box lacrosse.
└── Lacrosse → The specific discipline covered here.
    └── Discipline Type → Types of lacrosse games.
        ├── Field Lacrosse → Outdoor version played on a large field with 10 players per team.
        └── Box Lacrosse → Indoor variant played in an enclosed arena with 6 players per team.

        └── Competition/Event → Major events and professional or collegiate tournaments.
            ├── Name → e.g., NCAA Lacrosse Championship, World Lacrosse Championship, Major League Lacrosse.

            └── Tournament Type →
                ├── Invitational
                ├── Championship
                └── Exhibition Match

            ├── Season/Year → e.g., 2023, 2024.

            └── Venue → Stadiums or arenas hosting the games.
                ├── Name → e.g., Gillette Stadium, M&T Bank Stadium.
                └── Location → City, Country.

            └── Round/Match →
                ├── Group Stage
                ├── Quarterfinals
                ├── Semifinals
                └── Final

            └── Team/Club → Participating national or club teams.
                ├── Name → e.g., Team USA, Canada, Philadelphia Wings.

                └── Club/League →
                    ├── National Lacrosse League (NLL)
                    └── Premier Lacrosse League (PLL)

                └── Athlete → Individual players on the team.
                    ├── Name → e.g., Lyle Thompson, Tom Schreiber, Trevor Baptiste.

                    └── Role/Position →
                        ├── Attackman
                        ├── Midfielder
                        ├── Defenseman
                        └── Goalkeeper

                    └── Equipment/Brand/Model →
                        ├── Sticks → e.g., Warrior Evo, STX Stallion
                        ├── Helmets → e.g., Cascade XRS, Warrior Burn
                        └── Gloves/Pads → e.g., Maverik M5, Epoch Integra

                    └── Time/Record →
                        ├── Most Goals in a Game
                        ├── Fastest Shot
                        └── Most Saves

                └── Management →
                    ├── Head Coach
                    ├── Assistant Coach
                    └── General Manager

                └── Sponsors →
                    ├── Nike
                    ├── Adidas
                    └── Under Armour

                └── Ranking →
                    ├── World Ranking
                    └── NCAA Ranking

            └── Prize Money/Rewards →
                ├── Cash Prizes → e.g., $10,000
                └── Awards → e.g., Gold Medal, MVP Trophy

            └── Federation →
                ├── World Lacrosse
                └── NCAA (for college-level lacrosse)
            └── Broadcasting → Media outlets broadcasting games.
                ├── ESPN
                ├── TSN
                └── Sportsnet
```