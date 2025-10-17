<|begin_of_text|><|start_header_id|>system<|end_header_id|>
### ROLE:
You are a Cycling Research Specialist, an expert in analyzing professional cycling events and developing race-specific glossaries you can access the website to find the information and find the cyclist name who attend the event.
You have an in-depth understanding of cycling terminology, race dynamics, and the linguistic aspects of cycling events.

### TASK:
Your task is to update the taxonomy input that is given to you with Athlete Detail Section for the Event. You are given two inputs, {athlete_detail} and taxonomy input, the main objective is to update the taxonomy input's JSON structure to have the information extracted from {athlete_detail}.


### Instruction
You have to follow instruction below step by step:
1. **Extract**
      - Extract Taxonomy Structure and Information.
2. **Identify**
      - Identify section of athlete or team for team sport in TAXONOMY input.
3. **Extract Athlete Detail**
    - Extract athlete detail information in AthleteDetail.
4 **Update**
    - Update the athlete detail to athlete section on TAXONOMY input.
    - For example, if the name of the athlete found in AthleteDetail is "Lewis Hamilton", update the AthleteName field in as "Lewis Hamilton"


### STRICT GUIDELINE
1. You can only update the existing structure given to you, do not add or remove any fields.
2. Use the AthleteDetail information do not use the placeholder or information athlete detail section on TAXONOMY.
3. You are to keep the same JSON format as the input
4. when you return the output, it has to be the same format as the taxonomy input, example: if taxonomy input has 10 keys, you should return the same 10 keys. no more and no less.



### AthleteDetail
{athlete_detail}

### OUTPUT:
- return in JSON format no explanation.

```json
```
<|eot_id|>


```json
{
  "Motorsports": {
    "F1 Racing": {
      "Grand Prix": {
        "Formula 1 Louis Vuitton Australian Grand Prix 2025": {
          "OrganizationAuthority": "FIA (Fédération Internationale de l'Automobile)",
          "SeasonEdition": "2025",
          "TournamentFormat": "Single race event within the Formula 1 World Championship season",
          "Venue": {
            "Name": "Albert Park Circuit",
            "Location": "Melbourne, Australia",
            "CircuitType": "Street Circuit"
          },
          "Round": "Australian Grand Prix",
          "TeamPlayerGroup": {
            "Teams": [
              {
                "TeamName": "Mercedes-AMG Petronas F1 Team",
                "Management": {
                  "TeamPrincipal": "Toto Wolff"
                },
                "Sponsors": ["Petronas", "Mercedes-Benz", "INEOS"],
                "TeamRanking": "2nd in 2025 FIA Constructors' Championship after Australian GP",
                "Fanbase": "Mercedes F1 Fans",
                "Athletes": [
                  {
                    "AthleteName": "George Russell",
                    "RolePosition": "Driver",
                    "CarNumber": 63,
                    "Nationality": "United Kingdom",
                    "Equipment": "Mercedes F1 W16",
                    "AthleteRanking": "3rd in 2025 FIA Drivers' Championship after Australian GP"
                  },
                  {
                    "AthleteName": "Andrea Kimi Antonelli",
                    "RolePosition": "Driver",
                    "CarNumber": 12,
                    "Nationality": "Italy",
                    "Equipment": "Mercedes F1 W16",
                    "AthleteRanking": "4th in 2025 FIA Drivers' Championship after Australian GP"
                  }
                ]
              },
              {
                "TeamName": "Oracle Red Bull Racing",
                "Management": {
                  "TeamPrincipal": "Christian Horner"
                },
                "Sponsors": ["Oracle", "Honda", "Infiniti"],
                "TeamRanking": "3rd in 2025 FIA Constructors' Championship after Australian GP",
                "Fanbase": "Red Bull Racing Fans",
                "Athletes": [
                  {
                    "AthleteName": "Max Verstappen",
                    "RolePosition": "Driver",
                    "CarNumber": 1,
                    "Nationality": "Netherlands",
                    "Equipment": "Red Bull RB21 Honda RBPT",
                    "AthleteRanking": "2nd in 2025 FIA Drivers' Championship after Australian GP"
                  },
                  {
                    "AthleteName": "Liam Lawson",
                    "RolePosition": "Driver",
                    "CarNumber": 30,
                    "Nationality": "New Zealand",
                    "Equipment": "Red Bull RB21 Honda RBPT",
                    "AthleteRanking": "Not classified (DNF) in Australian GP"
                  }
                ]
              },
              {
                "TeamName": "Scuderia Ferrari",
                "Management": {
                  "TeamPrincipal": "Frédéric Vasseur"
                },
                "Sponsors": ["Shell", "Ray-Ban", "UPS"],
                "TeamRanking": "7th in 2025 FIA Constructors' Championship after Australian GP",
                "Fanbase": "Ferrari Fans",
                "Athletes": [
                  {
                    "AthleteName": "Charles Leclerc",
                    "RolePosition": "Driver",
                    "CarNumber": 16,
                    "Nationality": "Monaco",
                    "Equipment": "Ferrari SF-25",
                    "AthleteRanking": "8th in 2025 FIA Drivers' Championship after Australian GP"
                  },
                  {
                    "AthleteName": "Lewis Hamilton",
                    "RolePosition": "Driver",
                    "CarNumber": 44,
                    "Nationality": "United Kingdom",
                    "Equipment": "Ferrari SF-25",
                    "AthleteRanking": "10th in 2025 FIA Drivers' Championship after Australian GP"
                  }
                ]
              },
              {
                "TeamName": "McLaren F1 Team",
                "Management": {
                  "TeamPrincipal": "Andrea Stella"
                },
                "Sponsors": ["Dell Technologies", "Gulf Oil", "Darktrace"],
                "TeamRanking": "1st in 2025 FIA Constructors' Championship after Australian GP",
                "Fanbase": "McLaren Fans",
                "Athletes": [
                  {
                    "AthleteName": "Lando Norris",
                    "RolePosition": "Driver",
                    "CarNumber": 4,
                    "Nationality": "United Kingdom",
                    "Equipment": "McLaren MCL39 Mercedes F1 M16",
                    "AthleteRanking": "1st in 2025 FIA Drivers' Championship after Australian GP"
                  },
                  {
                    "AthleteName": "Oscar Piastri",
                    "RolePosition": "Driver",
                    "CarNumber": 81,
                    "Nationality": "Australia",
                    "Equipment": "McLaren MCL39 Mercedes F1 M16",
                    "AthleteRanking": "9th in 2025 FIA Drivers' Championship after Australian GP"
                  }
                ]
              },
              {
                "TeamName": "Williams Racing",
                "Management": {
                  "TeamPrincipal": "James Vowles"
                },
                "Sponsors": ["Atlassian", "Duracell", "Lavazza"],
                "TeamRanking": "4th in 2025 FIA Constructors' Championship after Australian GP",
                "Fanbase": "Williams Fans",
                "Athletes": [
                  {
                    "AthleteName": "Alexander Albon",
                    "RolePosition": "Driver",
                    "CarNumber": 23,
                    "Nationality": "Thailand",
                    "Equipment": "Williams FW47 Mercedes F1 M16",
                    "AthleteRanking": "5th in 2025 FIA Drivers' Championship after Australian GP"
                  },
                  {
                    "AthleteName": "Carlos Sainz Jr.",
                    "RolePosition": "Driver",
                    "CarNumber": 55,
                    "Nationality": "Spain",
                    "Equipment": "Williams FW47 Mercedes F1 M16",
                    "AthleteRanking": "Not classified (DNF) in Australian GP"
                  }
                ]
              },
              {
                "TeamName": "Aston Martin Aramco F1 Team",
                "Management": {
                  "TeamPrincipal": "Mike Krack"
                },
                "Sponsors": ["Aramco", "Cognizant", "BWT"],
                "TeamRanking": "5th in 2025 FIA Constructors' Championship after Australian GP",
                "Fanbase": "Aston Martin Fans",
                "Athletes": [
                  {
                    "AthleteName": "Fernando Alonso",
                    "RolePosition": "Driver",
                    "CarNumber": 14,
                    "Nationality": "Spain",
                    "Equipment": "Aston Martin AMR25 Mercedes F1 M16",
                    "AthleteRanking": "Not classified (DNF) in Australian GP"
                  },
                  {
                    "AthleteName": "Lance Stroll",
                    "RolePosition": "Driver",
                    "CarNumber": 18,
                    "Nationality": "Canada",
                    "Equipment": "Aston Martin AMR25 Mercedes F1 M16",
                    "AthleteRanking": "6th in 2025 FIA Drivers' Championship after Australian GP"
                  }
                ]
              },
              {
                "TeamName": "Kick Sauber F1 Team",
                "Management": {
                  "TeamPrincipal": "Frédéric Vasseur"
                },
                "Sponsors": ["Stake", "Ferrari", "Richard Mille"],
                "TeamRanking": "6th in 2025 FIA Constructors' Championship after Australian GP",
                "Fanbase": "Sauber Fans",
                "Athletes": [
                  {
                    "AthleteName": "Nico Hülkenberg",
                    "RolePosition": "Driver",
                    "CarNumber": 27,
                    "Nationality": "Germany",
                    "Equipment": "Kick Sauber C45 Ferrari 067",
                    "AthleteRanking": "7th in 2025 FIA Drivers' Championship after Australian GP"
                  },
                  {
                    "AthleteName": "Gabriel Bortoleto",
                    "RolePosition": "Driver",
                    "CarNumber": 5,
                    "Nationality": "Brazil",
                    "Equipment": "Kick Sauber C45 Ferrari 067",
                    "AthleteRanking": "Not classified (DNF) in Australian GP"
                  }
                ]
              },
              {
                "TeamName": "Haas F1 Team",
                "Management": {
                  "TeamPrincipal": "Guenther Steiner"
                },
                "Sponsors": ["MoneyGram", "Ferrari", "1&1"],
                "TeamRanking": "10th in 2025 FIA Constructors' Championship after Australian GP",
                "Fanbase": "Haas Fans",
                "Athletes": [
                  {
                    "AthleteName": "Esteban Ocon",
                    "RolePosition": "Driver",
                    "CarNumber": 31,
                    "Nationality": "France",
                    "Equipment": "Haas VF-25 Ferrari 067",
                    "AthleteRanking": "13th in 2025 FIA Drivers' Championship after Australian GP"
                  },
                  {
                    "AthleteName": "Oliver Bearman",
                    "RolePosition": "Driver",
                    "CarNumber": 87,
                    "Nationality": "United Kingdom",
                    "Equipment": "Haas VF-25 Ferrari 067",
                    "AthleteRanking": "14th in 2025 FIA Drivers' Championship after Australian GP"
                  }
                ]
              },
              {
                "TeamName": "BWT Alpine F1 Team",
                "Management": {
                  "TeamPrincipal": "Otmar Szafnauer"
                },
                "Sponsors": ["BWT", "Renault", "Castrol"],
                "TeamRanking": "8th in 2025 FIA Constructors' Championship after Australian GP",
                "Fanbase": "Alpine Fans",
                "Athletes": [
                  {
                    "AthleteName": "Pierre Gasly",
                    "RolePosition": "Driver",
                    "CarNumber": 10,
                    "Nationality": "France",
                    "Equipment": "Alpine A525 Renault E-Tech RE25",
                    "AthleteRanking": "11th in 2025 FIA Drivers' Championship after Australian GP"
                  },
                  {
                    "AthleteName": "Jack Doohan",
                    "RolePosition": "Driver",
                    "CarNumber": 7,
                    "Nationality": "Australia",
                    "Equipment": "Alpine A525 Renault E-Tech RE25",
                    "AthleteRanking": "Not classified (DNF) in Australian GP"
                  }
                ]
              },
              {
                "TeamName": "Visa Cash App Racing Bulls F1 Team",
                "Management": {
                  "TeamPrincipal": "Franz Tost"
                },
                "Sponsors": ["Visa", "Cash App", "Honda"],
                "TeamRanking": "9th in 2025 FIA Constructors' Championship after Australian GP",
                "Fanbase": "Racing Bulls Fans",
                "Athletes": [
                  {
                    "AthleteName": "Yuki Tsunoda",
                    "RolePosition": "Driver",
                    "CarNumber": 22,
                    "Nationality": "Japan",
                    "Equipment": "Racing Bulls VCARB 02 Honda RBPT",
                    "AthleteRanking": "12th in 2025 FIA Drivers' Championship after Australian GP"
                  },
                  {
                    "AthleteName": "Isack Hadjar",
                    "RolePosition": "Driver",
                    "CarNumber": 6,
                    "Nationality": "France",
                    "Equipment": "Racing Bulls VCARB 02 Honda RBPT",
                    "AthleteRanking": "Not classified (DNS) in Australian GP"
                  }
                ]
              }
            ]
          },
          "Referee": "Race Director and FIA Stewards",
          "PrizeMoneyRewards": "Prize money distributed according to FIA regulations",
          "Broadcast": ["Sky Sports F1", "ESPN", "F1 TV Pro"]
        }
      }
    }
  }
}
```