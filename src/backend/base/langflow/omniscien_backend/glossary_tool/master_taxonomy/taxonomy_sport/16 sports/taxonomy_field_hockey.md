# Field Hockey Taxonomy
```json
{
    "level 1": {
        "category": "Hockey Sports",
        "description": "such as field hockey, ice hockey and roller hockey."
    },
    "level 2": {
        "category": "Field Hockey",
        "description": "A widely played team sport where players use curved sticks to hit a hard ball into the opponent\u2019s goal on grass or artificial turf.",
        "reference_website": "https://www.topendsports.com/sport/list/hockey-field.htm"
    }
}

```

### Core Attributes:
- **Sport Category**: Hockey Sports, The general category of the sport that includes all types of hockey competitions and formats.
- **Sport**: Field Hockey, A widely played team sport where players use curved sticks to hit a hard ball into the opponent’s goal on grass or artificial turf.
- **Focused Subcategory**: Outdoor Field Hockey, The specific variant of field hockey played outdoors, typically with 11 players per side on a large field.
- **Discipline Type**: Competition formats, e.g., 11-a-side, 5-a-side.
- **Competition/Event**: Tournament names, e.g., FIH Hockey World Cup, FIH Pro League.
- **Season/Year**: Year the event occurs, e.g., 2023.
- **Stage/Match/Round**: Competition phases per day, e.g., Quarterfinals, Semifinals, Final.
- **Team/Club**: Participating teams, e.g., India, Netherlands, Germany, Australia.
  - **Athlete**: Player names, e.g., Manpreet Singh, Eva de Goede, Alexander Hendrickx.
    - **Role/Position**: Player positions, e.g., Forward, Midfielder, Defender, Goalkeeper.
    - **Equipment/Brand/Model**: Player gear, e.g., Hockey sticks like Adidas Stiff, Grays GX1000.
  - **Management**: Team officials, e.g., Team Manager, Head Coach, Assistant Coach.
  - **Sponsors**: Team sponsors, e.g., OBO, Asics, Vantage.
- **Venue**: Event locations, e.g., Kalinga Stadium (Bhubaneswar), The Wagener Stadium (Amsterdam).
- **Referee**: Match officials, e.g., FIH Umpires, International Umpires.
- **Ranking/Standings**: Team rankings and tournament results, e.g., World Rankings, Tournament Standings.


### Example Taxonomy Structure for Field Hockey:
```plaintext
Hockey Sports → Sport Category
A dynamic team sport played with sticks and a ball or puck. This taxonomy focuses on Field Hockey, one of the most widely played and internationally recognized variants.

└── Field Hockey → The specific hockey discipline highlighted here.
    └── Discipline Type → Game formats within field hockey.
        ├── 11-a-side → Standard full-team match format, 11 players per team.
        └── 5-a-side → A fast-paced version with 5 players per team, used in some leagues and youth competitions.

        └── Competition/Event → Organized international or domestic tournaments.
            ├── Name → e.g., FIH Hockey World Cup, FIH Pro League, Commonwealth Games, EuroHockey Championship.
            ├── Season/Year → e.g., 2023, 2024, 2025.

            └── Venue → Locations hosting the matches.
                ├── Stadium → e.g., Kalinga Stadium (Bhubaneswar), Wagener Stadium (Amsterdam), Lee Valley Hockey Centre (London).
                └── Country/City → e.g., Netherlands, India, Belgium, Argentina.

            └── Stage/Match/Round → Phases of the tournament.
                ├── Group Stage
                ├── Quarterfinals
                ├── Semifinals
                └── Final

            └── Team/Club → National or club teams participating in the competition.
                ├── National Teams → e.g., India, Netherlands, Germany, Australia.
                └── Clubs/Leagues → e.g., Bloemendaal (Netherlands), Dabang Mumbai (India), Ranchi Rays.

                └── Athlete → Players representing each team.
                    ├── Name → e.g., Manpreet Singh, Eva de Goede, Alexander Hendrickx.
                    ├── Role/Position →
                        • Forward – Offensive player, primary scorer.
                        • Midfielder – Link between attack and defense.
                        • Defender – Protects the goal area.
                        • Goalkeeper – Defends the goal, wears protective gear.

                    └── Equipment/Brand/Model → Gear used by athletes.
                        ├── Stick Models → e.g., Adidas Stiff, Grays GX1000, TK Total One.
                        ├── Protective Gear → e.g., Shin guards, gloves, helmets.
                        └── Shoes/Clothing → e.g., Asics, Adidas, Mizuno.

                ├── Management → Team staff and leadership.
                    ├── Head Coach
                    ├── Assistant Coach
                    ├── Team Manager
                    └── Medical Staff

                ├── Sponsors → Brands and companies supporting teams or events.
                    ├── Equipment Sponsors → e.g., OBO, Grays, Adidas.
                    ├── Apparel Sponsors → e.g., Asics, Vantage.
                    └── Corporate Sponsors → e.g., Hero, Odisha Tourism.

            └── Referee/Umpire → Match officials enforcing the rules.
                ├── FIH Umpires
                └── International Umpires

            └── Ranking/Standings → Performance statistics and rankings.
                ├── World Rankings → Maintained by FIH.
                └── Tournament Standings → Final rankings within an event.
```
