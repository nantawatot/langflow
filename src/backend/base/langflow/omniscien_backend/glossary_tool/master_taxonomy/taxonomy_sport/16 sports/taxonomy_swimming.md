# Swimming Taxonomy
```json
{
    "level 1": {
        "category": "Swimming",
        "description": "the sport of propelling oneself through water using the limbs."
    },
    "level 2": {
        "category": "Freestyle",
        "description": "in these events competitors can swim using any stroke of their choice.",
        "reference_website": "https://www.topendsports.com/sport/list/swimming-freestyle.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Swimming, The general category of the sport that includes all types of swimming competitions and formats.
- **Sport**: Swimming, The sport of propelling oneself through water using the limbs. It includes various strokes and distances.
- **Discipline Type**: Various swimming styles, such as Freestyle, Backstroke, Breaststroke, Butterfly, and Medley.
- **Competition/Event**: Event names, e.g., Olympic Games, FINA World Championships, World Cup Series.
- **Season/Year**: Year the event occurs, e.g., 2023.
- **Heat/Final**: Competition stages, e.g., Preliminary Heat, Semifinals, Final.
- **Athlete**: Athlete names, e.g., Michael Phelps, Katie Ledecky, Caeleb Dressel.
  - **Role/Position**: Athlete roles, e.g., World Record Holder, Olympic Champion.
  - **Team/Club**: Participating teams, e.g., USA Swimming, Swim Canada, Great Britain Swimming.
  - **Management**: Team officials, e.g., Coach, Team Manager, Sports Director.
  - **Sponsors**: Athlete sponsors, e.g., Speedo, Nike, Under Armour.
  - **Ranking**: Athlete rankings, e.g., FINA World Rankings.
  - **Equipment/Brand/Model**: Equipment used, e.g., Pools, Swim Caps, Swimwear such as Arena, Speedo Fastskin.

- **Venue**: Event locations, e.g., Tokyo Aquatics Centre, Beijing National Aquatics Center.
- **Prize Money/Rewards**: Prize amounts or awards, e.g., $1 million, Gold Medal.
- **Federation**: Organizing bodies, e.g., FINA (Fédération Internationale de Natation), National Swimming Federations.
- **Time/Record**: Performance stats, e.g., World Record, Personal Best.
- **Club/League**: Associated clubs or leagues, e.g., NCAA Swimming, ISL (International Swimming League).
- **Tournament Type**: Competition types, e.g., Invitational, Championship, Exhibition.
- **Relay/Team Event**: Team-based swimming events, e.g., 4x100m Freestyle Relay, 4x200m Freestyle Relay.
- **Open Water**: Long-distance swimming events in natural water bodies, e.g., 10km Marathon Swim, 5km Open Water Swim.
- **Para Swimming**: Swimming events for athletes with disabilities, e.g., S6 Freestyle, S8 Backstroke.
- **Broadcasting**: Media outlets broadcasting games, e.g., NBC Sports, Eurosport, Olympic Channel.

### Example Taxonomy Structure for Swimming:
```plaintext
Swimming → Sport Category
An individual or team water sport that involves racing using various stroke techniques in pools or open water.

└── Freestyle → A competitive swimming stroke that emphasizes speed and fluid motion, often the fastest stroke used in races.

    └── Competition/Event → Official swimming meets and tournaments.
        ├── Olympic Games
        ├── FINA World Championships
        ├── World Cup Series
        └── Continental Championships (e.g., European Championships, Pan Pacific Championships)

        └── Season/Year → e.g., 2023, 2024

        └── Venue →
            ├── Tokyo Aquatics Centre (Japan)
            ├── Beijing National Aquatics Center (China)
            ├── Budapest Duna Arena (Hungary)
            └── London Aquatics Centre (UK)

        └── Heat/Final → Phases of a swimming event.
            ├── Preliminary Heat
            ├── Semifinal
            └── Final

        └── Nation/Team → Participating national teams or federations.
            ├── USA Swimming
            ├── Swim Canada
            ├── Great Britain Swimming
            ├── Australian Dolphins Swim Team
            └── China Swimming Association

            └── Athlete → Competitive swimmers.
                ├── Michael Phelps
                ├── Katie Ledecky
                ├── Caeleb Dressel
                ├── Sarah Sjöström
                └── Sun Yang

                └── Role/Position →
                    ├── Olympic Champion
                    ├── World Record Holder
                    ├── National Champion
                    └── Relay Team Member

                └── Equipment/Brand/Model →
                    ├── Swim Cap (Arena, TYR, Speedo)
                    ├── Racing Suit (Speedo Fastskin, Arena Carbon Air)
                    ├── Goggles (Arena Cobra, Speedo Pure Focus)
                    └── Pool Technology (Omega Timing, Anti Wave Lanes)

                └── Ranking →
                    ├── FINA World Rankings
                    ├── National Rankings
                    └── World Cup Leaderboard

                └── Time/Record →
                    ├── World Record
                    ├── Olympic Record
                    ├── National Record
                    └── Personal Best

            └── Management →
                ├── Head Coach
                ├── Assistant Coach
                ├── Team Manager
                └── Sports Director

            └── Sponsors →
                ├── Speedo
                ├── Arena
                ├── TYR
                ├── Nike
                └── Under Armour

            └── Prize Money/Rewards →
                ├── Gold Medal
                ├── Silver Medal
                ├── Bronze Medal
                └── Cash Prizes

            └── Federation →
                ├── FINA (World Aquatics)
                ├── USA Swimming
                ├── Swimming Canada
                ├── British Swimming
                └── Australian Swimming Federation

```