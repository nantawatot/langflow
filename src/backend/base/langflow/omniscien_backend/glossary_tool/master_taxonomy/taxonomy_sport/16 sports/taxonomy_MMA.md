# Martial Arts Taxonomy
```json
{
    "level 1": {
        "category": "Martial Arts",
        "description": "such as the Olympic sports of judo, taekwondo and karate, and the multitude of wrestling sports."
    },
    "level 2": {
        "category": "MMA",
        "description": "a full-contact individual combat sport which include aspects of several other combat sports and martial arts.",
        "reference_website": "https://www.topendsports.com/sport/list/mma.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Martial Arts, The general category of combat sports that includes various fighting disciplines.
- **Sport**: MMA (Mixed Martial Arts), A full-contact individual combat sport which includes aspects of several other combat sports and martial arts. It is characterized by a combination of striking and grappling techniques, allowing fighters to use a wide range of skills and strategies.
- **Discipline Type**: Competition formats, e.g., Fight Night, Pay-Per-View, Championship Bout.
- **Competition/Event**: Event names, e.g., UFC 270, Bellator 245, ONE Championship.
- **Season/Year**: Year the event occurs, e.g., 2023.
- **Round/Match/Segment**: Competition rounds, e.g., Round 1, Round 2, Championship Round.
- **Fighter**: Athlete names, e.g., Conor McGregor, Khabib Nurmagomedov, Amanda Nunes.
  - **Role/Position**: Fighter roles, e.g., Fighter, Champion, Challenger.
  - **Weight Class**: Weight divisions, e.g., Lightweight, Welterweight, Heavyweight.
  - **Fight Style**: Fighting styles, e.g., Striker, Grappler, BJJ Specialist, Wrestler.
  - **Trainer/Coach**: Coaches and trainers, e.g., John Kavanagh, Mike Brown.
  - **Management**: Team officials, e.g., Manager, Promoter, Event Coordinator.
- **Sponsors**: Team sponsors, e.g., Reebok, Monster Energy, Harley-Davidson.
- **Referee**: Match officials, e.g., Herb Dean, Big John McCarthy.
- **Venue**: Event locations, e.g., T-Mobile Arena, Madison Square Garden, MGM Grand.
- **Title/Championship**: Championship titles, e.g., UFC Lightweight Championship, Bellator Featherweight Title.
- **Broadcasting**: Media outlets broadcasting events, e.g., ESPN, Fox Sports, DAZN.
- **Fight Promotion**: Organizing bodies, e.g., UFC, Bellator, ONE Championship, Rizin Fighting Federation.
- **Fight Style**: Fighting styles, e.g., Striker, Grappler, Brazilian Jiu-Jitsu Specialist, Wrestler.


### Example Taxonomy Structure for MMA:
```plaintext
Martial Arts → Sport Category
A category of combat sports focused on self-defense, discipline, and full-contact competition.

└── MMA (Mixed Martial Arts) → A modern full-contact sport combining techniques from various martial arts disciplines.

    └── Discipline Type → Types of organized MMA events.
        ├── Fight Night → Regular events with ranked or upcoming fighters.
        ├── Pay-Per-View (PPV) → Major fights aired for purchase, often featuring title bouts.
        └── Championship Bout → Fights determining or defending titles.

        └── Competition/Event → Official fight events.
            ├── Name → e.g., UFC 270, Bellator 245, ONE Fight Night 10.

            └── Title/Championship →
                ├── UFC Lightweight Championship
                ├── Bellator Featherweight Title
                └── ONE Flyweight World Championship

            └── Season/Year → e.g., 2023, 2024

            └── Venue →
                ├── Name → e.g., T-Mobile Arena, Madison Square Garden, MGM Grand
                └── Location → City, Country

            └── Round/Match/Segment →
                ├── Round 1
                ├── Round 2
                ├── Championship Round
                └── Extra/Overtime Round (if applicable)

            └── Fighter → Combatants in each bout.
                ├── Name → e.g., Conor McGregor, Amanda Nunes, Khabib Nurmagomedov

                └── Role/Position →
                    ├── Fighter
                    ├── Champion
                    └── Challenger

                └── Weight Class →
                    ├── Flyweight
                    ├── Bantamweight
                    ├── Featherweight
                    ├── Lightweight
                    ├── Welterweight
                    ├── Middleweight
                    ├── Light Heavyweight
                    └── Heavyweight

                └── Fight Style →
                    ├── Striker
                    ├── Grappler
                    ├── Wrestler
                    └── Brazilian Jiu-Jitsu (BJJ) Specialist

                └── Trainer/Coach →
                    ├── John Kavanagh
                    ├── Mike Brown
                    └── Javier Mendez

                └── Management →
                    ├── Manager
                    ├── Promoter
                    └── Event Coordinator

                └── Sponsors →
                    ├── Reebok
                    ├── Monster Energy
                    ├── Crypto.com
                    └── Harley-Davidson

            └── Referee →
                ├── Herb Dean
                ├── Big John McCarthy
                └── Marc Goddard

            └── Broadcasting →
                ├── ESPN
                ├── Fox Sports
                └── DAZN

            └── Fight Promotion →
                ├── UFC (Ultimate Fighting Championship)
                ├── Bellator MMA
                ├── ONE Championship
                └── Rizin Fighting Federation
```