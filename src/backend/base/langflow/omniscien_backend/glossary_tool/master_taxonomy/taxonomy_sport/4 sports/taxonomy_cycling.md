# Cycling Sports Taxonomy
```json
{
    "level 1": {
        "category": "Cycling Sports",
        "description": "sports involving bicycles."
    },
    "level 2": {
        "category": "Road Cycling",
        "description": "cycle races held on paved roads, usually over several hours or days.",
        "reference_website": "https://www.topendsports.com/sport/list/cycling-road.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Cycling Sports, The general category of the sport that includes all types of competitive cycling disciplines such as road cycling, track cycling, mountain biking, BMX, and more.
- **Sport**: Road Cycling, A specialized sub-discipline of cycling that focuses on races conducted on paved roads. It includes individual and team races that range from single-day classics to multi-stage events like the Tour de France.
- **Discipline Type**: The format or structure of the road cycling event. Common types include:
    - Stage Race – A race spanning multiple days with cumulative times (e.g., Tour de France).
    - Time Trial – Riders race individually or in teams against the clock.
    - One-day Race – A race completed in a single day (e.g., Paris–Roubaix).
- **Competition/Event**: The official name of a The official name of a race or cycling event, often sanctioned by a governing body (like UCI). Examples include world-renowned events such as the Tour de France, Giro d’Italia, and Vuelta a España.
- **Season/Year**: The year in which the competition takes place. This helps organize event editions chronologically, e.g., Tour de France 2023.
- **Stage/Match/Round**: A specific daily race segment in a multi-day event. Each stage may vary in terrain and difficulty (e.g., flat stage, mountain stage, time trial) and is identified as Stage 1, Stage 2, etc.
- **Team/Club**: A professional or amateur cycling team registered to compete in events. Each team is composed of athletes, support staff, and management. Examples: Jumbo-Visma, UAE Team Emirates.
   - **Athlete**: A professional rider who competes for a team in races. Cyclists often have specialized roles within the team. Examples: Jonas Vingegaard, Tadej Pogačar.
   - **Role/Position**: The functional role or specialization of an athlete within a team. These include:
      - Climber – Excels in mountain stages.
      - Sprinter – Strong in fast, flat finishes.
      - All-rounder – Versatile across stage types.
      - Domestique – Supports team leaders during races.
   - **Ranking**: The athlete's position in the UCI World Ranking, which reflects their performance across events.
   - **BIB Number**: A unique identifier assigned to each athlete for the event,
   - **Management**: The leadership team responsible for overseeing the riders and race strategy. This includes roles like:
      - Team Manager – Oversees overall team operations.
      - Sport Director – Manages tactical decisions during races.
- **Organization/Authority**: The governing body that oversees the rules and regulations of cycling events. The Union Cycliste Internationale (UCI) is the primary authority for international cycling competitions.
- **Sponsors**: Companies or organizations that provide financial or material support to teams. Sponsors typically appear in team names and branding. Examples: Jumbo, Visma, UAE, Emirates.
- **Broadcast**: The media outlets responsible for televising or streaming the event, such as ESPN or Eurosport.


### Example Taxonomy Structure for Cycling Sports:
```plaintext
Cycling Sports → The overarching sport category that includes all forms of competitive cycling, such as road racing, mountain biking, track cycling, BMX, and more.
└── Road Cycling → A discipline of cycling that takes place on paved roads. It includes single-day races, time trials, and multi-stage tours, and is the most globally recognized form of cycling competition.

    └── Discipline Type → The type or structure of the race format used in road cycling.
        ├── Stage Race → Multi-day events with cumulative timing (e.g., Tour de France).
        ├── Time Trial → Riders race individually or as a team against the clock.
        └── One-day Race → A single race completed within one day (e.g., Milan–San Remo).

            └── Competition/Event → The official name of a cycling race or tournament (e.g., Tour de France, Giro d’Italia).
                ├── Season/Year → The year the event takes place (e.g., 2025).
                ├── Venue → The location or host cities of the event (e.g., Paris, Milan, Barcelona).

                ├── Stage/Match/Round → A segment or phase of a race in a multi-stage event.
                │   ├── Stage Number → Identifier (e.g., Stage 1, Stage 2).
                │   ├── Date → The specific date of the stage.
                │   ├── Route Profile → The nature of the route (e.g., flat, hilly, mountain, time trial).
                │   ├── Distance → Length of the stage (e.g., 155 km).
                │   └── Start/Finish Locations → Cities or towns where the stage starts and ends.

                ├── Team/Club → A registered professional or amateur cycling team participating in the event.
                │   ├── Team Name → Official name of the team (e.g., Jumbo-Visma, UAE Team Emirates).
                │   ├── Management → Team leadership (e.g., Team Manager, Sport Director).
                │   ├── Sponsors → Supporting brands/organizations (e.g., Jumbo, Visma).
                │   └── Athletes → Cyclists representing the team.
                │       ├── Athlete Name → Full name of the cyclist (e.g., Jonas Vingegaard).
                │       ├── Role/Position → Athlete’s specialization (e.g., Climber, Sprinter, Domestique).
                │       ├── Ranking → UCI World Ranking position (e.g., No. 1).
                │       └── BIB Number → Unique rider number for the event (e.g., BIB 101).

                ├── Organization/Authority → Governing body for the event (e.g., UCI - Union Cycliste Internationale).
                ├── Prize Money/Rewards → Financial and symbolic rewards for performance (e.g., €500,000 for 1st place).
                └── Broadcast → TV and online platforms covering the event (e.g., Eurosport, NBC Sports, FloBikes).

```