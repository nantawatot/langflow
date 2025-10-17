# Boxing Taxonomy
```json
{
    "level 1": {
        "category": "Martial Arts",
        "description": "such as the Olympic sports of judo, taekwondo and karate, and the multitude of wrestling sports."
    },
    "level 2": {
        "category": "Boxing",
        "description": "a combat sport in which two players throw punches at each other.",
        "reference_website": "https://www.topendsports.com/sport/list/boxing.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Martial Arts, The general category of combat sports that includes various fighting disciplines.
- **Sport**: Boxing, A combat sport where two competitors fight in a ring using punches with gloved hands, following specific rules and regulations. Boxing is a popular martial art emphasizing physical strength, technique, and strategy.
- **Level**: Classification of boxing, e.g., Professional Boxing, Amateur Boxing.
- **Discipline Type**: Competition formats, e.g., Title Fight, Non-title Fight, Exhibition Match.
- **Competition/Event**: Tournament or event names, e.g., WBC World Heavyweight Championship, Boxing Day Fight Night, Golden Gloves.
- **Venue/Location**: Event locations, e.g., Madison Square Garden, MGM Grand Garden Arena, T-Mobile Arena.
- **Season/Year**: Year the event occurs, e.g., 2023, 2024.
  - Main event Boxing Match: The primary boxing match or championship event.
    - Rule/Regulation: Specific rules governing the match, such as weight classes, rounds, and scoring.
    - ฺBoxing Round: Specific rounds of the match, e.g., 3 Rounds, 5 Rounds, 12 Rounds.
    - Athlete: Boxer names, e.g., Tyson Fury, Canelo Alvarez, Muhammad Ali.
      - Team/Club: Promoters or organizations involved, e.g., Top Rank, Matchroom Boxing, PBC (Premier Boxing Champions).
      - Nationality: Boxer’s country of origin, e.g., USA, UK, Mexico.
      - Ranking: Boxer rankings, e.g., WBC Heavyweight Champion, IBF Middleweight Champion.
      - Coach: The coach or trainer of the boxer, e.g., Freddie Roach, Eddie Reynoso.
      - Record: Boxer’s fight record, e.g., Wins, Losses, Draws.
  - Preliminary Fights: Under-card matches leading up to the main event.
    - Rule/Regulation: Specific rules for preliminary fights, such as shorter rounds or different weight classes.
    - Boxing Round: Specific rounds of the preliminary fights, e.g., 3 Rounds, 4 Rounds.
    - Athlete: Boxer names in preliminary fights, e.g., Up-and-coming boxers, Local fighters.
      - Team/Club: Promoters or organizations involved in the preliminary fights, e.g., Local Boxing Promotions, Regional Boxing Clubs.
      - Nationality: Boxer’s country of origin, e.g., USA, UK, Mexico.
      - Ranking: Boxer rankings in preliminary fights, e.g., Regional Champion, National Title Holder.
      - Coach: The coach or trainer of the boxer in preliminary fights, e.g., Local Coaches, Regional Trainers.
      - Record: Boxer’s fight record in preliminary fights, e.g., Wins, Losses, Draws.

- **Sponsors**: Sponsors supporting athletes and events, e.g., Adidas, Under Armour, Everlast, Hennessy, MGM Grand.
- **Title/Championship**: Championship titles or sanctioning bodies, e.g., WBA, WBC, IBF, WBO, Unified Champion.
- **Judge/Referee**: Officials overseeing matches, e.g., Referee, Ringside Judges.
- **Federation**: Governing bodies, e.g., WBC (World Boxing Council), WBA (World Boxing Association), IBF (International Boxing Federation).
  - President: The head of the boxing federation, e.g., Mauricio Sulaiman (WBC), Gilberto Mendoza (WBA).
  - National Federations: Country-specific boxing organizations, e.g., USA Boxing, British Boxing Board of Control.
- **Broadcasting**: Media outlets airing the matches, e.g., ESPN, DAZN, Showtime Boxing.



### Example Taxonomy Structure for Boxing:

```plaintext
Martial Arts → Sport Category
Combat sports that focus on fighting techniques, physical strength, and strategy. Boxing is a popular martial art emphasizing punches with gloved hands in a regulated ring.
├── Boxing → The specific martial arts discipline focused on here, involving two competitors fighting in a ring using punches within set rules.

│   ├── Level → Competition tier or professional status, such as Professional Boxing or Amateur Boxing.
│   │   ├── Discipline Type → Formats of boxing matches, including:
│   │       • Title Fight – Contests where a championship belt is at stake.
│   │       • Non-title Fight – Regular bouts without championship implications.
│   │       • Exhibition Match – Non-competitive or charity bouts often for entertainment.

│   │   │   ├── Competition/Event → Named boxing events or championships, e.g., WBC World Heavyweight Championship, Boxing Day Fight Night, Golden Gloves.
│   │   │   │   ├── Season/Year → The year or season when the event occurs (e.g., 2023, 2024).
│   │   │   │   ├── Venue/Location → The arenas or venues hosting the matches, such as Madison Square Garden, MGM Grand Garden Arena, T-Mobile Arena.
│   │   │   │   ├── Main Event Boxing Match → The primary boxing match or championship event.
│   │   │   │   │   ├── Rule/Regulation → Specific rules governing the match, such as weight classes, rounds, and scoring.
│   │   │   │   │   │   │   ├── Boxing Round → Specific rounds of the match
│   │   │   │   │   │   │   ├── 3 Rounds → Shorter matches, often in preliminary fights.
│   │   │   │   │   │   │   ├── 5 Rounds → Common for non-title fights.
│   │   │   │   │   │   │   └── 12 Rounds → Standard for championship bouts.

│   │   │   │   │   ├── Athlete → Boxer names, e.g., Tyson Fury, Canelo Alvarez, Muhammad Ali.
│   │   │   │   │   │   ├── Team/Club → Promoters or organizations involved, e.g., Top Rank, Matchroom Boxing, PBC (Premier Boxing Champions).
│   │   │   │   │   │   ├── Nationality → Boxer’s country of origin, e.g., USA, UK, Mexico.
│   │   │   │   │   │   ├── Ranking → Boxer rankings, e.g., WBC Heavyweight Champion, IBF Middleweight Champion.
│   │   │   │   │   │   ├── Coach → The coach or trainer of the boxer, e.g., Freddie Roach, Eddie Reynoso.
│   │   │   │   │   │   └── Record → Boxer’s fight record, e.g., Wins, Losses, Draws.

│   │   │   │   ├── Preliminary Fights → Under-card matches leading up to the main event.
│   │   │   │   │   ├── Rule/Regulation → Specific rules for preliminary fights, such as shorter rounds or different weight classes.
│   │   │   │   │   │   ├── Boxing Round → Specific rounds of the preliminary fights, e.g., 3 Rounds, 4 Rounds.
│   │   │   │   │   ├── Athlete → Boxer names in preliminary fights, e.g., Up-and-coming boxers, Local fighters.
│   │   │   │   │   │   ├── Team/Club → Promoters or organizations involved in the preliminary fights, e.g., Local Boxing Promotions, Regional Boxing Clubs.
│   │   │   │   │   │   ├── Nationality → Boxer’s country of origin, e.g., USA, UK, Mexico.
│   │   │   │   │   │   ├── Ranking → Boxer rankings in preliminary fights, e.g., Regional Champion, National Title Holder.
│   │   │   │   │   │   ├── Coach → The coach or trainer of the boxer in preliminary fights, e.g., Local Coaches, Regional Trainers.
│   │   │   │   │   │   └── Record → Boxer’s fight record in preliminary fights, e.g., Wins, Losses, Draws.

│   │   │   │   ├── Sponsors → Brands supporting athletes and events, e.g., Adidas, Under Armour, Everlast, Hennessy, MGM Grand.
│   │   │   │   ├── Title/Championship → Championship titles or sanctioning bodies, e.g., WBA, WBC, IBF, WBO, Unified Champion.
│   │   │   │   ├── Judge/Referee → Officials overseeing matches, e.g., Referee, Ringside Judges.
│   │   │   │   ├── Federation → Governing bodies, e.g., WBC (World Boxing Council), WBA (World Boxing Association), IBF (International Boxing Federation).
│   │   │   │   │   ├── President → The head of the boxing federation, e.g., Mauricio Sulaiman (WBC), Gilberto Mendoza (WBA).
│   │   │   │   │   └── National Federations → Country-specific boxing organizations, e.g., USA Boxing, British Boxing Board of Control.
│   │   │   │   ├── Broadcasting → Media outlets airing the matches, e.g., ESPN, DAZN, Showtime Boxing.

```








