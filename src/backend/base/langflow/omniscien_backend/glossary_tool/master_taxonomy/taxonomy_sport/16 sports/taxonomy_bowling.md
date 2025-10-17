# Bowling Taxonomy
```json
{
    "level 1": {
        "category": "Bowling Sports",
        "description": "there are many variations of bowling or throwing a ball to knock down pins or get close to a target, such as the popular 10-pin bowling and boules."
    },
    "level 2": {
        "category": "Tenpin",
        "description": "a player rolls a bowling ball along a wooden or synthetic lane to knock down pins.",
        "reference_website": "https://www.topendsports.com/sport/list/bowling-tenpin.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Bowling Sports, The general category of the sport that includes all types of bowling competitions and formats.
- **Sport**: Subcategory of Bowling Sport:
  - **Tenpin**: The most recognized form of bowling, played with ten pins arranged in a triangular formation and a heavy ball rolled down a lane.
  - **Nine-pin**: A variation of bowling played with nine pins, often seen in European countries.
- **Discipline Type**: Competition formats, e.g., Singles, Doubles, Team Event.
- **Competition/Event**: Named bowling tournaments or tours, e.g., PBA Tour, World Bowling Championship.
- **Season/Year**: The year or season in which the event occurs, e.g., 2023.
- **Match/Round**: Competition stages, e.g., Round 1, Quarterfinal, Final.
- **Team/Club**: Participating teams or clubs, e.g., Team USA, Brunswick Pro Bowling.
  - **Athlete**: Player names, e.g., Jason Belmonte, Sean Rash, Mika Koivuniemi.
    - **Role/Position**: Player roles within a team, such as Lead Bowler, Anchor, Spare Specialist.
    - **Title/Championship**: Names of titles or awards held, e.g., PBA World Championship, World Cup of Bowling.
    - **Equipment/Brand/Model**: Bowling gear used by athletes, including bowling balls, shoes, bags (e.g., Storm IQ Tour, Dexter SST).
  - **Management**: Team officials such as Coach, Manager, Team Director.
  - **Sponsors**: Brands supporting teams or players, e.g., Storm, Brunswick, Ebonite.
- **Referee**: Officials overseeing competition fairness, e.g., John Smith, Mark Johnson.
- **Broadcasting**: Media channels airing the event, such as ESPN, Fox Sports.
- **Association/Federation**: Governing bodies organizing competitions, e.g., Professional Bowlers Association (PBA), World Bowling Federation.
- **Tournament Format**: Types of competition structures, including Elimination, Round-robin, Knockout.


### Example Taxonomy Structure for Bowling:
```plaintext
Bowling → Sport Category
A target sport where players roll a ball down a lane to knock down pins. Includes variations, with Tenpin bowling being the most common internationally.

├── Tenpin → Specific bowling format with 10 pins and heavy balls on synthetic lanes.
└── Nine-pin → A variation of bowling played with nine pins, often seen in European countries.

    └── Discipline Type → Competitive formats within Tenpin bowling.
        ├── Singles → Individual players compete head-to-head or against the field.
        ├── Doubles → Two-player teams compete in combined scoring formats.
        └── Team Event → Groups of players (typically 4–6) compete for team scores.

        └── Competition/Event → Official tournaments or leagues.
            ├── Name → e.g., PBA Tour, World Bowling Championship, USBC Open Championships.
            ├── Season/Year → Calendar year of the competition (e.g., 2024).

            └── Venue → Bowling centers or arenas where matches are held.
                ├── Bowling Center → e.g., Thunderbowl Lanes, South Point Bowling Plaza.
                └── Location → City or country (e.g., Las Vegas, Tokyo, Helsinki).

            └── Match/Round → Tournament stages or brackets.
                ├── Round 1
                ├── Quarterfinal
                ├── Semifinal
                └── Final

            └── Team/Club → Organized teams or national squads.
                ├── Name → e.g., Team USA, Brunswick Pro Bowling, Storm Nation.

                └── Athlete → Competitors in the event.
                    ├── Name → e.g., Jason Belmonte, Sean Rash, Shannon O'Keefe.
                    ├── Role/Position → Player's function in team settings:
                        • Lead Bowler, Anchor, Spare Specialist.
                    ├── Title/Championship → Honors won:
                        • PBA World Championship, World Cup of Bowling, Player of the Year.

                    └── Equipment/Brand/Model → Bowling gear used.
                        ├── Ball → e.g., Storm IQ Tour, Hammer Black Widow.
                        ├── Shoes → e.g., Dexter SST 8, Brunswick TPU-X.
                        └── Accessories → e.g., wrist guards, bags, grip tapes.

                ├── Management → Team leadership and support staff.
                    ├── Head Coach
                    ├── Assistant Coach
                    └── Team Manager/Director

                ├── Sponsors → Brands backing the athlete or team.
                    ├── Ball Manufacturers → e.g., Storm, Ebonite, Brunswick.
                    ├── Apparel/Shoe Brands → e.g., Dexter, 3G.
                    └── Other Sponsors → e.g., Coca-Cola, Monster Energy.

            └── Referee → Officials ensuring rules and scoring integrity.
                ├── Name → e.g., John Smith, Mark Johnson.
                └── Role → Lane official, scoring supervisor, tournament judge.

            └── Broadcasting → Media outlets and streaming platforms.
                ├── TV Network → e.g., ESPN, Fox Sports.
                └── Online Streaming → e.g., BowlTV, YouTube, FloBowling.

            └── Association/Federation → Governing bodies organizing events.
                ├── Professional Bowlers Association (PBA)
                ├── World Bowling Federation
                └── United States Bowling Congress (USBC)

            └── Tournament Format → Structure of competitive play.
                ├── Elimination → Loser is removed from bracket.
                ├── Round-robin → Everyone competes with every other participant.
                └── Knockout → Progressive elimination to determine final winner.
```