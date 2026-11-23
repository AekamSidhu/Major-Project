"""Calibrate the echo detector against modern LLM interview answers.

Parakeet AI (and similar tools) let candidates choose GPT-5, GPT-4.1,
or Claude 4.0 Sonnet. This script builds a calibration set of:
  - AI answers written in the style those models produce
  - Real human answers (natural, conversational, imperfect)
and runs them through the echo model to check whether the thresholds
still separate the two in an interview context.

Usage:
    python training/echo-detection/calibrate_interview.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "local-ai"))
from integrity.models import EchoModel

# ── Interview questions and paired answers ──────────────────────────────
# Each entry: (question, human_answer, ai_answer)
# Human answers: natural speech patterns, filler, personal anecdotes, imperfect structure.
# AI answers: GPT/Claude style — structured, comprehensive, slightly formal, uses
#   transition words, hedging phrases, and thorough coverage.

CALIBRATION_DATA = [
    # -- Behavioral / general --
    (
        "Tell me about yourself.",
        # Human
        "So I did my undergrad in computer science at University of Waterloo, graduated last year. "
        "I've been working at a startup since then doing mostly backend stuff, Python and Go. "
        "Before that I interned at Shopify for like eight months which was really cool, I got to work "
        "on their payments team. Honestly I just really like building things that people actually use, "
        "you know? That's what gets me up in the morning.",
        # AI (GPT-style)
        "I'm a software engineer with a strong foundation in backend development and distributed systems. "
        "I hold a Bachelor's degree in Computer Science from the University of Waterloo, where I developed "
        "a passion for building scalable, reliable software solutions. Currently, I work at a fast-growing "
        "startup where I design and implement microservices using Python and Go. Prior to this, I completed "
        "an eight-month internship at Shopify, where I contributed to the payments infrastructure team. "
        "This experience gave me valuable exposure to high-throughput transaction processing and taught me "
        "the importance of writing robust, well-tested code. I'm particularly drawn to roles where I can "
        "leverage my technical skills to solve complex problems and deliver meaningful impact.",
    ),
    (
        "What is your greatest strength?",
        # Human
        "I'd say I'm pretty good at debugging. Like when something breaks in production, I don't panic, "
        "I just start narrowing things down systematically. My manager actually mentioned this in my last "
        "review, she said I was the person the team comes to when stuff hits the fan. I think it's because "
        "I'm naturally curious and I like figuring out puzzles, even when they're stressful.",
        # AI
        "My greatest strength is my analytical problem-solving ability, particularly when it comes to "
        "debugging complex systems. I have a methodical approach to identifying root causes — I systematically "
        "eliminate potential issues by analyzing logs, tracing execution paths, and isolating variables. "
        "This skill has been recognized by my team and leadership; during performance reviews, I've been "
        "highlighted as the go-to person for resolving critical production issues. I believe this strength "
        "stems from my inherent curiosity and my commitment to understanding systems at a fundamental level, "
        "which allows me to remain calm and effective even under pressure.",
    ),
    (
        "Why do you want to work here?",
        # Human
        "Honestly I've been following your blog for a while and the technical challenges you write about "
        "are really interesting. The post about migrating to Kubernetes while keeping zero downtime was "
        "nuts. Also I talked to Sarah who works on your platform team at a meetup and she had great things "
        "to say about the culture. I want to work somewhere where I can learn from people who are better "
        "than me, and it seems like this is that kind of place.",
        # AI
        "I'm excited about this opportunity for several compelling reasons. First, I've been an avid follower "
        "of your engineering blog, and the technical challenges your team tackles — such as the recent "
        "zero-downtime Kubernetes migration — align perfectly with my interests in distributed systems and "
        "infrastructure. Second, through networking events, I've had the opportunity to connect with current "
        "team members who speak highly of the collaborative engineering culture. I'm looking for an environment "
        "where I can both contribute meaningfully and continue to grow as an engineer, and your company's "
        "commitment to technical excellence and knowledge sharing makes it an ideal fit for my career goals.",
    ),
    (
        "Tell me about a time you failed.",
        # Human
        "Oh man, so last year I pushed a database migration that I didn't test properly and it took down "
        "our checkout flow for like 45 minutes during peak hours. It was bad. I'd skipped the staging "
        "environment because I thought the change was simple enough. After that I was pretty embarrassed "
        "but I wrote up a postmortem and proposed we add a migration checklist that the whole team uses now. "
        "Learned the hard way that there's no such thing as a simple migration.",
        # AI
        "A significant learning experience occurred when I deployed a database migration that caused a "
        "45-minute outage of our checkout flow during peak traffic hours. The root cause was my decision "
        "to bypass our staging environment, as I had assessed the change as low-risk. This experience taught "
        "me several valuable lessons. First, I learned that no change should be considered too simple for "
        "proper testing. Second, I took immediate accountability by conducting a thorough postmortem analysis "
        "and proposing a comprehensive migration checklist. This checklist has since been adopted by the "
        "entire team and has prevented similar incidents. The experience reinforced my belief in the importance "
        "of robust processes and humility in engineering.",
    ),
    (
        "Where do you see yourself in five years?",
        # Human
        "That's a tough one. I think I want to get into more of a tech lead role eventually. I've been "
        "mentoring a couple junior devs on my team and I really enjoy that side of things. But I also don't "
        "want to stop coding, you know? So maybe like a senior engineer who also leads a small team. "
        "I'm not super interested in going full management, at least not yet.",
        # AI
        "In five years, I envision myself in a technical leadership role where I can combine my engineering "
        "expertise with team mentorship. I'm particularly interested in growing into a senior engineer or "
        "tech lead position where I can architect solutions, guide technical decisions, and mentor junior "
        "developers. I've already begun developing these skills through mentoring two junior engineers on "
        "my current team, which has been incredibly rewarding. I value the hands-on technical work and want "
        "to maintain a strong coding practice while also contributing to team growth and strategic direction. "
        "I believe this balanced approach to technical leadership aligns well with the growth opportunities "
        "at your organization.",
    ),
    (
        "How do you handle conflict with a coworker?",
        # Human
        "I had a situation where me and another dev disagreed about how to structure an API. He wanted REST, "
        "I wanted GraphQL. It got kind of heated in a meeting honestly. After that I just asked if we could "
        "grab coffee and talk it through. Turned out we were both worried about different things — he cared "
        "about caching and I cared about the frontend having to make too many requests. We ended up doing "
        "REST with some batch endpoints which actually worked great.",
        # AI
        "I approach workplace conflicts with open communication and a focus on finding mutually beneficial "
        "solutions. For example, I once had a disagreement with a colleague regarding API architecture — "
        "I advocated for GraphQL while they preferred REST. When the discussion became unproductive in a "
        "group setting, I took the initiative to schedule a one-on-one conversation in a more relaxed "
        "environment. Through this dialogue, I discovered that our concerns were complementary rather than "
        "contradictory: they were focused on caching efficiency while I was concerned about minimizing "
        "frontend request overhead. By understanding each other's perspectives, we developed a hybrid "
        "solution using REST with batch endpoints that addressed both concerns effectively. This experience "
        "reinforced my belief that most conflicts stem from miscommunication rather than genuine disagreement.",
    ),
    (
        "What's your biggest weakness?",
        # Human
        "I tend to overthink things sometimes. Like I'll spend way too long designing something perfectly "
        "before I start building it. My tech lead had to tell me a couple times to just start coding and "
        "iterate. I've gotten better at it, I try to set time limits for myself now, like okay you have "
        "30 minutes to design this then start building. But yeah it's still something I work on.",
        # AI
        "One area I've been actively working to improve is my tendency toward over-engineering solutions. "
        "I sometimes invest excessive time in upfront design and architecture, striving for an optimal "
        "solution before writing any code. While thoroughness is generally valuable, I've recognized that "
        "this can sometimes delay delivery and limit opportunities for iterative learning. To address this, "
        "I've implemented several strategies: I now set explicit time-boxes for design phases, I practice "
        "building minimum viable implementations first, and I've embraced the principle of iterative "
        "refinement. My tech lead has noted significant improvement in my ability to balance planning with "
        "execution, and I continue to refine this balance.",
    ),
    (
        "Describe a project you're proud of.",
        # Human
        "So at my last job we had this internal tool for managing deployments that everyone hated. It was "
        "this old PHP thing that barely worked. I convinced my manager to let me rebuild it over a few "
        "sprints. Used React and a simple Go backend. The best part was I did user interviews with the "
        "other engineers first so I actually built what they needed. After launch our deployment time "
        "went from like 20 minutes of clicking around to under 5 minutes. People were really happy with it "
        "and that felt awesome.",
        # AI
        "I'm particularly proud of a deployment management tool I developed at my previous company. The "
        "existing solution was a legacy PHP application that caused significant friction in the deployment "
        "process. I identified this as an opportunity for improvement and successfully advocated for a "
        "modernization initiative spanning several sprints. My approach began with conducting user research "
        "— I interviewed engineers across the organization to understand their pain points and requirements. "
        "Based on these insights, I architected and built a modern solution using React for the frontend "
        "and Go for the backend. The results were substantial: deployment time decreased from approximately "
        "20 minutes to under 5 minutes, representing a 75% reduction in time spent on deployments. The tool "
        "received overwhelmingly positive feedback from the engineering team and continues to be a critical "
        "part of their infrastructure.",
    ),
    # -- Technical --
    (
        "Explain the difference between a process and a thread.",
        # Human
        "So a process is like its own thing, it has its own memory space and everything. A thread is "
        "lighter, it runs inside a process and shares memory with the other threads in that process. "
        "That's why threads are faster to create and switch between but also why you can get race conditions "
        "and stuff like that. Processes are more isolated which is safer but they're heavier. Like if you "
        "fork a process you're basically copying the whole thing. At least that's how it works on Linux.",
        # AI
        "A process and a thread are both units of execution, but they differ significantly in several key "
        "aspects. A process is an independent execution environment with its own dedicated memory space, "
        "file descriptors, and system resources. It provides strong isolation, meaning one process cannot "
        "directly access another process's memory. Processes communicate through inter-process communication "
        "mechanisms such as pipes, sockets, or shared memory segments. A thread, on the other hand, is a "
        "lightweight unit of execution that exists within a process. Multiple threads within the same process "
        "share the same memory space and resources, which makes them more efficient for concurrent operations "
        "but introduces challenges such as race conditions, deadlocks, and the need for synchronization "
        "primitives like mutexes and semaphores. Context switching between threads is faster than between "
        "processes due to the shared memory space. The choice between processes and threads depends on the "
        "specific requirements: processes offer better isolation and fault tolerance, while threads provide "
        "lower overhead and easier data sharing.",
    ),
    (
        "What is a RESTful API?",
        # Human
        "REST is basically a way to structure your web APIs. You use HTTP methods like GET, POST, PUT, "
        "DELETE to do CRUD operations on resources. Each resource has a URL, like slash users slash 123 "
        "would get user 123. It's stateless so every request has everything the server needs. I mean "
        "that's the theory, in practice a lot of APIs call themselves RESTful but aren't really. "
        "Like they don't use proper status codes or they put verbs in the URL.",
        # AI
        "A RESTful API, or Representational State Transfer API, is an architectural style for designing "
        "networked applications. It adheres to several core principles that make it scalable, maintainable, "
        "and intuitive. First, it uses standard HTTP methods — GET for retrieval, POST for creation, PUT "
        "and PATCH for updates, and DELETE for removal — to perform operations on resources. Second, "
        "resources are identified by unique URIs, such as /users/123 for a specific user. Third, the "
        "communication is stateless, meaning each request contains all the information needed for the "
        "server to process it, without relying on stored session state. Additional principles include a "
        "uniform interface, layered system architecture, and the ability to cache responses for improved "
        "performance. RESTful APIs are widely adopted due to their simplicity, platform independence, and "
        "alignment with the HTTP protocol. They are commonly used for building web services, mobile "
        "application backends, and microservice architectures.",
    ),
    (
        "How would you optimize a slow database query?",
        # Human
        "First thing I'd do is look at the query plan, like EXPLAIN ANALYZE in Postgres. That usually "
        "tells you right away if it's doing a sequential scan when it should be using an index. Then "
        "I'd check if the right indexes exist. After that it depends — maybe the query is doing too many "
        "joins, maybe the table needs vacuuming, maybe you're selecting columns you don't need. I had one "
        "case where adding a composite index on two columns the WHERE clause used took a query from "
        "8 seconds to 20 milliseconds. Sometimes you also need to denormalize if joins are killing you.",
        # AI
        "Optimizing a slow database query involves a systematic approach with several key strategies. "
        "First, I would analyze the query execution plan using tools like EXPLAIN ANALYZE in PostgreSQL "
        "to identify bottlenecks such as sequential scans, inefficient joins, or suboptimal index usage. "
        "Based on this analysis, I would consider the following optimizations: Adding or modifying indexes "
        "— including composite indexes for multi-column WHERE clauses and covering indexes to avoid table "
        "lookups. Query restructuring — rewriting subqueries as JOINs, eliminating unnecessary columns in "
        "SELECT statements, and breaking complex queries into simpler components. Schema optimization — "
        "considering denormalization for read-heavy workloads, implementing proper partitioning for large "
        "tables, and ensuring appropriate data types are used. Database maintenance — running VACUUM and "
        "ANALYZE to update statistics and reclaim storage. Caching strategies — implementing application-level "
        "caching with tools like Redis for frequently accessed data. Additionally, I would consider read "
        "replicas for distributing query load and connection pooling to manage database connections efficiently. "
        "The specific approach depends on the nature of the query and the broader system architecture.",
    ),
    (
        "What is the CAP theorem?",
        # Human
        "It's about distributed systems. Basically it says you can only have two out of three things: "
        "consistency, availability, and partition tolerance. And since network partitions always happen "
        "in real systems, you're really choosing between consistency and availability. Like Postgres gives "
        "you consistency, DynamoDB gives you availability. Though honestly it's more nuanced than that, "
        "it's not really a binary choice. Most systems let you tune it per operation.",
        # AI
        "The CAP theorem, formulated by computer scientist Eric Brewer, is a fundamental principle in "
        "distributed systems design. It states that a distributed data store can simultaneously guarantee "
        "at most two of the following three properties: Consistency — every read receives the most recent "
        "write or an error, ensuring all nodes see the same data at the same time. Availability — every "
        "request receives a non-error response, without guarantee that it contains the most recent write. "
        "Partition Tolerance — the system continues to operate despite network partitions between nodes. "
        "Since network partitions are inevitable in distributed systems, the practical choice is typically "
        "between consistency and availability. CP systems, such as traditional relational databases like "
        "PostgreSQL, prioritize consistency and partition tolerance, potentially sacrificing availability "
        "during partitions. AP systems, such as Amazon DynamoDB and Apache Cassandra, prioritize availability "
        "and partition tolerance, accepting eventual consistency. Modern distributed systems often implement "
        "tunable consistency levels, allowing developers to make granular trade-offs based on specific "
        "use case requirements.",
    ),
    (
        "What's the difference between SQL and NoSQL databases?",
        # Human
        "SQL databases are relational, so you have tables with rows and columns and you define a schema "
        "upfront. Postgres, MySQL, that kind of thing. NoSQL is everything else basically — document stores "
        "like MongoDB, key-value stores like Redis, graph databases like Neo4j. The main tradeoff is SQL "
        "gives you ACID transactions and strong consistency but it's harder to scale horizontally. NoSQL "
        "is more flexible with schema and scales better across servers but you give up some of those "
        "guarantees. We used MongoDB at my last job for the product catalog because the schema kept changing "
        "and it worked well for that.",
        # AI
        "SQL and NoSQL databases represent fundamentally different approaches to data storage and management, "
        "each with distinct advantages. SQL databases are relational database management systems that store "
        "data in structured tables with predefined schemas. They use Structured Query Language for querying "
        "and support ACID transactions, ensuring data integrity. Examples include PostgreSQL, MySQL, and "
        "Oracle. NoSQL databases encompass a variety of non-relational database types: document stores like "
        "MongoDB, key-value stores like Redis, column-family stores like Cassandra, and graph databases like "
        "Neo4j. They offer flexible schemas, horizontal scalability, and are optimized for specific access "
        "patterns. Key differences include schema flexibility — NoSQL allows dynamic schemas while SQL "
        "requires predefined structures. Scalability — SQL databases typically scale vertically while NoSQL "
        "databases are designed for horizontal scaling. Consistency — SQL databases provide strong ACID "
        "guarantees while many NoSQL databases offer eventual consistency. The choice between SQL and NoSQL "
        "depends on factors such as data structure, scalability requirements, consistency needs, and the "
        "specific use case. Many modern architectures employ a polyglot persistence approach, using both "
        "SQL and NoSQL databases for different components of the system.",
    ),
    (
        "How does HTTPS work?",
        # Human
        "So when you go to an HTTPS site your browser and the server do a TLS handshake. The server sends "
        "its certificate, the browser checks it's signed by a CA it trusts. Then they agree on a symmetric "
        "key using something like Diffie-Hellman and after that everything is encrypted. It's not encrypting "
        "with the certificate's public key the whole time because that'd be way too slow — asymmetric crypto "
        "is just for the handshake part.",
        # AI
        "HTTPS, or Hypertext Transfer Protocol Secure, works by layering HTTP on top of TLS (Transport Layer "
        "Security) to provide encrypted communication. The process involves several key steps. First, the "
        "TLS handshake initiates when a client connects to a server. The server presents its digital "
        "certificate, which contains its public key and is signed by a trusted Certificate Authority. The "
        "client verifies the certificate's validity by checking the CA's signature, the certificate's "
        "expiration date, and that the domain matches. Next, the client and server negotiate encryption "
        "parameters and perform a key exchange, typically using algorithms like Diffie-Hellman or ECDHE, to "
        "establish a shared symmetric session key. This approach is used because symmetric encryption is "
        "significantly more computationally efficient than asymmetric encryption for data transfer. Once "
        "the session key is established, all subsequent communication is encrypted using symmetric encryption "
        "algorithms such as AES. This ensures confidentiality, integrity, and authentication of data "
        "transmitted between the client and server.",
    ),
    (
        "What are microservices?",
        # Human
        "Instead of having one big application you break it into smaller services that each do one thing. "
        "Like you'd have a user service, an order service, a payment service. They talk to each other "
        "over HTTP or message queues. The nice thing is each team can deploy their service independently "
        "and use whatever language they want. The downside is it's way more complex — you need service "
        "discovery, distributed tracing, handling failures between services. At my company we started "
        "with a monolith and only split out services when we actually needed to scale them separately.",
        # AI
        "Microservices architecture is an approach to software development where an application is structured "
        "as a collection of loosely coupled, independently deployable services. Each microservice is designed "
        "around a specific business capability and can be developed, deployed, and scaled independently. Key "
        "characteristics include single responsibility — each service focuses on one business function, such "
        "as user management, order processing, or payment handling. Independent deployment — services can be "
        "updated and deployed without affecting other services. Technology diversity — teams can choose the "
        "most appropriate technology stack for each service. Communication — services interact through "
        "well-defined APIs, typically using REST, gRPC, or message queues. Microservices offer benefits such "
        "as improved scalability, faster development cycles, better fault isolation, and organizational "
        "alignment with cross-functional teams. However, they also introduce complexity in areas such as "
        "service discovery, distributed data management, inter-service communication, monitoring, and "
        "debugging. Many organizations adopt a pragmatic approach, starting with a monolithic architecture "
        "and decomposing into microservices as the application and team grow.",
    ),
    (
        "Explain polymorphism in object-oriented programming.",
        # Human
        "Polymorphism is when you can treat different objects the same way. Like if you have a Shape class "
        "and Circle and Rectangle extend it, you can call draw on any of them and each one does its own "
        "thing. There's compile-time polymorphism which is method overloading and runtime polymorphism which "
        "is method overriding. I use it all the time, like you have an interface and different implementations "
        "and the calling code doesn't care which one it gets.",
        # AI
        "Polymorphism is a fundamental concept in object-oriented programming that allows objects of different "
        "types to be treated through a uniform interface. The term derives from Greek, meaning 'many forms.' "
        "There are two primary types of polymorphism. Compile-time polymorphism, also known as static "
        "polymorphism, is achieved through method overloading, where multiple methods share the same name "
        "but differ in parameters. The appropriate method is determined at compile time based on the argument "
        "types. Runtime polymorphism, also known as dynamic polymorphism, is achieved through method "
        "overriding and inheritance. A subclass provides its own implementation of a method defined in its "
        "parent class, and the specific implementation is determined at runtime based on the actual object "
        "type. For example, a Shape base class might define a draw method, with Circle and Rectangle "
        "subclasses each providing their own implementation. Code that operates on Shape references can call "
        "draw without knowing the specific subtype. Polymorphism enables several benefits: code reusability, "
        "extensibility, and adherence to the Open/Closed Principle, allowing systems to be extended without "
        "modifying existing code.",
    ),
    (
        "What is Docker and why would you use it?",
        # Human
        "Docker is basically containers. You package your app with all its dependencies into an image and "
        "then you can run it anywhere. So instead of the classic 'works on my machine' problem, everyone "
        "runs the same container. We use it at work for local dev so everyone has the same Postgres version "
        "and everything. And then in production we run containers on Kubernetes. It's also great for CI "
        "because you spin up a clean environment every time.",
        # AI
        "Docker is a containerization platform that enables developers to package applications along with "
        "all their dependencies, libraries, and configuration files into standardized units called containers. "
        "These containers are lightweight, portable, and consistent across different environments. Key "
        "benefits of using Docker include environment consistency — Docker eliminates the 'it works on my "
        "machine' problem by ensuring identical runtime environments across development, testing, and "
        "production. Isolation — each container runs in its own isolated environment, preventing dependency "
        "conflicts between applications. Portability — containers can run on any system that supports Docker, "
        "regardless of the underlying infrastructure. Efficiency — containers share the host OS kernel, "
        "making them more lightweight than traditional virtual machines. Scalability — Docker integrates "
        "seamlessly with orchestration tools like Kubernetes for managing containerized applications at "
        "scale. Common use cases include standardizing development environments, implementing CI/CD "
        "pipelines with clean build environments, deploying microservices architectures, and simplifying "
        "application distribution. Docker has become an essential tool in modern software development and "
        "DevOps practices.",
    ),
    (
        "How do you ensure code quality?",
        # Human
        "Code reviews are the big one for me. Having another person look at your code catches so much. "
        "We also have a CI pipeline that runs tests and linting on every PR. I try to write tests as I go, "
        "not after the fact. And honestly just keeping functions small and naming things well goes a long way. "
        "Oh and we do pair programming sometimes for tricky stuff, that's been super helpful.",
        # AI
        "Ensuring code quality requires a multi-faceted approach encompassing several practices and tools. "
        "Code reviews are fundamental — peer reviews catch bugs, improve design, and facilitate knowledge "
        "sharing across the team. I prioritize thorough, constructive reviews that focus on correctness, "
        "maintainability, and adherence to coding standards. Automated testing is essential, including unit "
        "tests for individual components, integration tests for system interactions, and end-to-end tests "
        "for critical user flows. I advocate for writing tests alongside feature development rather than "
        "retroactively. Continuous integration pipelines automate quality checks on every pull request, "
        "including test execution, static analysis, linting, and code coverage measurement. Clean code "
        "practices such as meaningful naming conventions, single-responsibility functions, consistent "
        "formatting, and comprehensive documentation contribute to long-term maintainability. Additionally, "
        "pair programming for complex implementations, regular refactoring to address technical debt, "
        "and architectural decision records for significant design choices all support sustained code quality.",
    ),
    (
        "What's the difference between authentication and authorization?",
        # Human
        "Authentication is proving who you are — like logging in with your username and password. "
        "Authorization is what you're allowed to do after that. So authentication is 'are you really John' "
        "and authorization is 'can John access this admin page.' A lot of people mix them up but they're "
        "separate concerns. We use JWTs for auth at my company, the token has the user's roles in it and "
        "then middleware checks if they have the right role for each endpoint.",
        # AI
        "Authentication and authorization are two distinct but complementary security concepts that are often "
        "confused. Authentication is the process of verifying a user's identity — confirming that they are "
        "who they claim to be. Common authentication mechanisms include username/password combinations, "
        "multi-factor authentication, biometric verification, OAuth tokens, and certificate-based "
        "authentication. It answers the question 'Who are you?' Authorization is the process of determining "
        "what an authenticated user is permitted to do — what resources they can access and what operations "
        "they can perform. Authorization mechanisms include role-based access control, attribute-based access "
        "control, and permission-based systems. It answers the question 'What are you allowed to do?' In "
        "practice, authentication always precedes authorization — you must first verify identity before "
        "determining permissions. A common implementation pattern uses JSON Web Tokens, where the "
        "authentication process generates a token containing the user's identity and roles, and middleware "
        "checks these roles against endpoint-specific access requirements for authorization decisions.",
    ),
    # -- Situational / soft skills --
    (
        "How do you prioritize when you have multiple deadlines?",
        # Human
        "I make a list first and figure out what's actually urgent vs what just feels urgent. Usually I talk "
        "to my manager or the stakeholders to understand which things really can't slip. Then I try to "
        "estimate how long each thing will take and see if the math works out. If it doesn't I flag it "
        "early rather than trying to do everything and doing a bad job. I learned the hard way that "
        "saying 'I can't do all of these by Friday' is way better than delivering half-baked work.",
        # AI
        "When faced with multiple competing deadlines, I employ a structured prioritization framework. "
        "First, I assess each task based on urgency and impact, distinguishing between truly time-sensitive "
        "requirements and perceived urgency. I engage in proactive communication with stakeholders and "
        "management to understand business priorities and negotiate realistic timelines when necessary. "
        "I estimate the effort required for each task and evaluate whether the collective workload is "
        "feasible within the given timeframes. If the analysis reveals an unrealistic workload, I escalate "
        "early with specific recommendations — such as suggesting scope reductions, timeline adjustments, "
        "or additional resource allocation. I've found that transparent communication about capacity "
        "constraints leads to better outcomes than attempting to deliver everything and compromising quality. "
        "Tools like task boards and time-blocking help me maintain focus and track progress effectively.",
    ),
    (
        "Tell me about a time you had to learn something quickly.",
        # Human
        "When I joined my current team they were using Rust for some performance-critical services and I'd "
        "never written Rust before. I had to fix a bug in one of those services within my first two weeks. "
        "I basically spent every evening going through the Rust book and writing little programs. I also "
        "bugged my teammate Jake a lot, he was really patient about explaining ownership and borrowing. "
        "I got the bug fixed on day 10 and honestly the forced learning under pressure was great, I actually "
        "really like Rust now.",
        # AI
        "A notable example of rapid learning occurred when I joined a team that utilized Rust for "
        "performance-critical microservices — a language I had no prior experience with. Within my first "
        "two weeks, I was tasked with resolving a bug in one of these Rust services. I developed an "
        "accelerated learning plan that included systematic study of the official Rust documentation, "
        "particularly focusing on the ownership and borrowing model. I supplemented this with hands-on "
        "practice by writing small programs to reinforce concepts. Additionally, I leveraged the expertise "
        "of experienced team members who provided valuable mentorship on Rust-specific patterns and idioms. "
        "Through this intensive approach, I successfully diagnosed and resolved the bug within ten days. "
        "This experience reinforced my ability to learn rapidly under pressure and demonstrated the "
        "effectiveness of combining structured self-study with collaborative learning. I've since become "
        "proficient in Rust and actively contribute to the team's Rust codebase.",
    ),
    (
        "How do you stay up to date with technology?",
        # Human
        "I read Hacker News most days, though I try not to get sucked into the comments too much. I follow "
        "some people on Twitter who post good stuff. And I have a few newsletters I actually read like "
        "TLDR and Pragmatic Engineer. For deeper stuff I'll watch conference talks on YouTube, like I "
        "watched all the GopherCon talks last year. I also just build side projects when something looks "
        "interesting, that's the best way to actually learn something.",
        # AI
        "I maintain a comprehensive approach to staying current with technological developments. I regularly "
        "follow industry news aggregators like Hacker News and curated newsletters such as TLDR and The "
        "Pragmatic Engineer for high-level awareness. For deeper technical knowledge, I watch conference "
        "presentations from events like GopherCon, KubeCon, and Strange Loop, which provide insights from "
        "industry practitioners. I maintain an active presence on technical communities and social media "
        "platforms where thought leaders share insights and emerging trends. Perhaps most importantly, I "
        "engage in hands-on learning through personal projects that allow me to explore new technologies "
        "in a practical context. I believe experiential learning is the most effective way to truly "
        "understand a technology. Additionally, I participate in technical book clubs with colleagues and "
        "contribute to open-source projects, which expose me to diverse codebases and development practices.",
    ),
    (
        "What would you do if you disagreed with a technical decision your team made?",
        # Human
        "Depends on how big the decision is I guess. If it's small I'd probably just go with it, not "
        "everything is worth fighting over. If it's something that could really hurt us I'd bring it up, "
        "but I'd make sure I had actual reasons and not just 'I don't like it.' I'd probably write up my "
        "concerns with examples and data if I had any. And then if the team still decides the other way, "
        "I commit to it. You can't just keep relitigating decisions or nothing gets done.",
        # AI
        "I believe in constructive disagreement and would handle this situation with both conviction and "
        "respect. First, I would assess the significance of the decision — not every disagreement warrants "
        "escalation, and I recognize that perfect can be the enemy of good. For significant technical "
        "decisions where I have genuine concerns, I would prepare a thoughtful analysis documenting my "
        "perspective with concrete evidence, potential risks, and alternative approaches. I would present "
        "this in an appropriate forum, focusing on technical merits rather than personal preferences. I would "
        "actively listen to the team's reasoning, as they may have context or constraints I'm not aware of. "
        "If the team ultimately decides on a different approach after considering my input, I fully commit "
        "to that decision and support its implementation. I believe in the principle of 'disagree and commit' "
        "— maintaining team cohesion and momentum is essential, and relitigating resolved decisions "
        "undermines productivity and trust.",
    ),
    (
        "Why are you leaving your current job?",
        # Human
        "I've been there almost three years now and I feel like I've kind of plateaued. The tech stack "
        "hasn't changed much and the team is small so there's not much room to grow. My manager is great "
        "and I'll miss the people honestly, but I want to work on harder problems and learn from more "
        "senior engineers. I'm not running away from anything, I just feel ready for the next challenge.",
        # AI
        "I'm seeking a new opportunity because I've reached a natural inflection point in my career growth "
        "at my current company. After nearly three years, I've developed strong expertise in my current "
        "domain but have limited opportunities for further advancement due to the team's size and the "
        "relatively stable technology stack. I value my current colleagues and leadership greatly, and my "
        "departure is motivated entirely by a desire for growth rather than dissatisfaction. Specifically, "
        "I'm looking for an environment that offers more complex technical challenges, exposure to larger-scale "
        "systems, and the opportunity to learn from and collaborate with senior engineers who can accelerate "
        "my professional development. I believe this role aligns well with these growth objectives.",
    ),
    # -- More technical depth --
    (
        "Explain how garbage collection works.",
        # Human
        "So garbage collection is automatic memory management. The runtime keeps track of which objects "
        "are still reachable from your code and frees the ones that aren't. In Java it uses a generational "
        "approach where new objects go in the young generation and get collected frequently, and ones that "
        "survive get promoted to the old generation. The tricky part is stop-the-world pauses where the GC "
        "has to freeze your app. Modern collectors like G1 and ZGC try to minimize that. Go's GC is "
        "concurrent and has really short pauses, that's one of the things I like about Go.",
        # AI
        "Garbage collection is an automatic memory management technique that identifies and reclaims memory "
        "occupied by objects that are no longer referenced by the application. The fundamental process involves "
        "several phases. Root identification — the collector identifies root references, which include stack "
        "variables, static fields, and CPU registers. Reachability analysis — starting from roots, the "
        "collector traverses the object graph, marking all reachable objects as live. Reclamation — "
        "unreachable objects are identified as garbage, and their memory is freed for reuse. Most modern "
        "garbage collectors employ a generational approach based on the generational hypothesis — the "
        "observation that most objects die young. Memory is divided into generations: a young generation "
        "for newly created objects, collected frequently using minor collections, and an old generation for "
        "long-lived objects, collected less frequently using major collections. Advanced collectors like "
        "Java's G1 and ZGC, and Go's concurrent collector, minimize stop-the-world pauses through concurrent "
        "marking and incremental collection strategies. Each language and runtime makes different trade-offs "
        "between throughput, latency, and memory overhead in their garbage collection implementations.",
    ),
    (
        "What is CI/CD and why is it important?",
        # Human
        "CI is continuous integration — basically every time you push code it gets built and tested "
        "automatically. CD is continuous delivery or deployment, where code that passes all checks gets "
        "pushed to production automatically or with one click. It's important because without it you end "
        "up with these painful manual release processes where things break because someone forgot a step. "
        "We use GitHub Actions at work and it's been great. Every PR runs tests, lint, and type checking, "
        "and once it's merged to main it deploys to staging automatically.",
        # AI
        "CI/CD, standing for Continuous Integration and Continuous Delivery/Deployment, is a set of practices "
        "and automation tools that form the backbone of modern software development workflows. Continuous "
        "Integration involves automatically building and testing code changes whenever developers push to "
        "the shared repository. This practice catches integration issues early, maintains code quality "
        "through automated test execution, and provides rapid feedback to developers. Continuous Delivery "
        "extends CI by automatically preparing code changes for release to production, ensuring that the "
        "codebase is always in a deployable state. Continuous Deployment goes further by automatically "
        "deploying every change that passes all pipeline stages directly to production. The importance of "
        "CI/CD includes faster feedback loops — developers receive immediate notification of build failures "
        "or test regressions. Reduced risk — smaller, more frequent deployments are inherently less risky "
        "than large, infrequent releases. Consistency — automated pipelines eliminate human error in build "
        "and deployment processes. Velocity — teams can deliver features and fixes to users more rapidly. "
        "Common CI/CD platforms include GitHub Actions, Jenkins, GitLab CI, and CircleCI.",
    ),
    (
        "How would you design a URL shortener?",
        # Human
        "I'd keep it simple. You need a database table mapping short codes to long URLs. When someone "
        "submits a URL, generate a random short code or use a base62 encoding of an auto-increment ID. "
        "Store the mapping. On redirect, look up the code and do a 301 or 302 redirect. For high traffic "
        "you'd put a cache like Redis in front. Oh and you need to handle collisions if you're doing random "
        "codes. That's basically it for an MVP, you could add analytics, custom codes, expiration later.",
        # AI
        "Designing a URL shortener involves addressing several key components and considerations. At its "
        "core, the system needs to generate unique short codes and map them to original URLs. For the "
        "URL shortening service, I would implement a code generation strategy using either Base62 encoding "
        "of an auto-incrementing ID for guaranteed uniqueness, or a hash-based approach with collision "
        "resolution. The data model would consist of a mapping table with columns for the short code, "
        "original URL, creation timestamp, expiration date, and analytics metadata. For the architecture, "
        "I would design a web service that handles two primary operations: creation through a POST endpoint "
        "that accepts a long URL and returns a shortened URL, and redirection through a GET endpoint that "
        "resolves the short code and performs an HTTP redirect. For scalability, I would implement a caching "
        "layer using Redis to store frequently accessed mappings, reducing database load. The database could "
        "be horizontally sharded by short code for distributed storage. Rate limiting would prevent abuse. "
        "Additional features would include custom short codes, URL expiration, click analytics with geographic "
        "and temporal data, and an API for programmatic access. For high availability, I would deploy across "
        "multiple regions with database replication.",
    ),
    (
        "What is event-driven architecture?",
        # Human
        "Instead of services calling each other directly, they publish events and other services subscribe "
        "to them. Like when an order is placed, the order service publishes an 'order created' event and "
        "then the payment service, inventory service, and email service each pick it up and do their thing. "
        "It's nice because the services are decoupled — the order service doesn't need to know about all "
        "the downstream stuff. We use Kafka for this at work. The downside is it's harder to debug because "
        "you can't just follow one request through the system.",
        # AI
        "Event-driven architecture is a software design pattern in which the flow of the system is determined "
        "by events — significant state changes or occurrences that components produce and consume. In this "
        "architecture, services communicate through events rather than direct synchronous calls, promoting "
        "loose coupling and scalability. The core components include event producers that generate events when "
        "state changes occur, event channels or message brokers like Apache Kafka, RabbitMQ, or AWS SNS/SQS "
        "that transport events reliably, and event consumers that subscribe to and process relevant events. "
        "For example, when an order is placed, the order service publishes an 'OrderCreated' event. The "
        "payment service, inventory service, and notification service each independently consume this event "
        "and perform their respective operations. Key benefits include loose coupling between services, "
        "improved scalability through asynchronous processing, enhanced resilience as failures in one consumer "
        "don't affect others, and easier extensibility since new consumers can be added without modifying "
        "existing services. Challenges include ensuring eventual consistency, handling event ordering, "
        "implementing idempotent consumers, and increased complexity in debugging and monitoring distributed "
        "event flows.",
    ),
    (
        "Explain the SOLID principles.",
        # Human
        "SOLID is five principles for object-oriented design. S is single responsibility — a class should "
        "do one thing. O is open-closed — open for extension, closed for modification. L is Liskov "
        "substitution — subtypes should be usable in place of their base type. I is interface segregation — "
        "don't make clients depend on methods they don't use. D is dependency inversion — depend on "
        "abstractions not concrete implementations. In practice I think SRP and DIP are the most useful "
        "ones. The others I apply when it makes sense but I don't obsess over them.",
        # AI
        "The SOLID principles are five foundational design principles for creating maintainable, scalable, "
        "and robust object-oriented software. Single Responsibility Principle states that a class should have "
        "only one reason to change, meaning it should encapsulate only one piece of functionality. This "
        "promotes cohesion and makes code easier to understand, test, and maintain. Open/Closed Principle "
        "states that software entities should be open for extension but closed for modification. New "
        "functionality should be added through inheritance or composition rather than altering existing code. "
        "Liskov Substitution Principle states that objects of a superclass should be replaceable with objects "
        "of its subclasses without affecting the correctness of the program. This ensures proper inheritance "
        "hierarchies. Interface Segregation Principle states that clients should not be forced to depend on "
        "interfaces they don't use. Smaller, focused interfaces are preferable to large, monolithic ones. "
        "Dependency Inversion Principle states that high-level modules should depend on abstractions rather "
        "than concrete implementations. This facilitates loose coupling and makes systems more testable and "
        "flexible. Together, these principles guide developers in creating software that is resilient to "
        "change, easy to maintain, and straightforward to extend.",
    ),
    # -- Claude-style AI answers (slightly different voice from GPT) --
    (
        "What interests you about backend development?",
        # Human
        "I like that the stuff I build powers everything else. The frontend is what people see but the "
        "backend is what makes it all work. There's something satisfying about building a system that "
        "handles thousands of requests per second and doesn't fall over. And I love the problem-solving "
        "aspect, like figuring out how to make a query efficient or how to handle race conditions. It's "
        "like solving puzzles all day and getting paid for it.",
        # AI (Claude-style)
        "What draws me to backend development is the combination of architectural thinking and practical "
        "problem-solving it demands. There's something deeply satisfying about designing systems that "
        "reliably handle high throughput while maintaining data integrity. I find the challenges inherently "
        "engaging — whether it's optimizing database queries, designing efficient caching strategies, or "
        "reasoning about concurrency and race conditions. Backend development also appeals to my interest "
        "in understanding systems at a fundamental level. The work involves navigating real constraints "
        "around performance, scalability, and reliability, and finding elegant solutions within those "
        "constraints. I appreciate that backend systems, while often invisible to end users, form the "
        "critical foundation that enables everything else to function correctly.",
    ),
    (
        "How do you approach debugging a production issue?",
        # Human
        "First thing is don't panic. Then I check the monitoring — are there error spikes, latency changes, "
        "anything obvious in the dashboards? Then I look at logs around the time it started. Usually there's "
        "a recent deploy that correlates. If I can reproduce it locally that's ideal but production bugs "
        "are often about scale or specific data so sometimes you can't. I keep narrowing it down — is it "
        "the database, the network, the code? Binary search basically. And I communicate with the team "
        "the whole time so people know what's going on.",
        # AI (Claude-style)
        "My approach to debugging production issues follows a systematic process that balances urgency with "
        "thoroughness. I start by assessing the scope and impact — understanding what's affected helps "
        "prioritize the response appropriately. Then I examine monitoring dashboards and alerting data to "
        "identify patterns: error rate spikes, latency changes, or resource utilization anomalies that "
        "correlate with the onset of the issue. I review recent deployments and configuration changes, as "
        "these are frequently the trigger. Log analysis comes next, focusing on the timeframe around when "
        "the issue began. I use a process of progressive narrowing — systematically eliminating possible "
        "causes to isolate the root cause. Is it database-related, network-related, or application logic? "
        "Throughout this process, I maintain clear communication with the team, providing regular updates "
        "on findings and next steps. Once resolved, I conduct a blameless postmortem to document the root "
        "cause, the resolution, and preventive measures to reduce the likelihood of recurrence.",
    ),
]


def main():
    model = EchoModel()
    print(f"Model: {model.name}")
    print(f"Current thresholds: possible={model.possible:.4f}, high={model.high:.4f}")
    print(f"Number of calibration pairs: {len(CALIBRATION_DATA)}")
    print()

    human_scores = []
    ai_scores = []

    for question, human_answer, ai_answer in CALIBRATION_DATA:
        h_result = model.assess(human_answer)
        a_result = model.assess(ai_answer)
        human_scores.append(h_result["ai_probability"])
        ai_scores.append(a_result["ai_probability"])

    import numpy as np

    human_scores = np.array(human_scores)
    ai_scores = np.array(ai_scores)

    print("=" * 70)
    print("HUMAN ANSWERS (should score LOW)")
    print("=" * 70)
    print(f"  Mean:   {human_scores.mean():.4f}")
    print(f"  Median: {np.median(human_scores):.4f}")
    print(f"  Std:    {human_scores.std():.4f}")
    print(f"  Min:    {human_scores.min():.4f}")
    print(f"  Max:    {human_scores.max():.4f}")
    print(f"  >possible ({model.possible:.3f}): {(human_scores >= model.possible).sum()}/{len(human_scores)}")
    print(f"  >high ({model.high:.3f}):     {(human_scores >= model.high).sum()}/{len(human_scores)}")
    print()

    print("=" * 70)
    print("AI ANSWERS (should score HIGH)")
    print("=" * 70)
    print(f"  Mean:   {ai_scores.mean():.4f}")
    print(f"  Median: {np.median(ai_scores):.4f}")
    print(f"  Std:    {ai_scores.std():.4f}")
    print(f"  Min:    {ai_scores.min():.4f}")
    print(f"  Max:    {ai_scores.max():.4f}")
    print(f"  >possible ({model.possible:.3f}): {(ai_scores >= model.possible).sum()}/{len(ai_scores)}")
    print(f"  >high ({model.high:.3f}):     {(ai_scores >= model.high).sum()}/{len(ai_scores)}")
    print()

    print("=" * 70)
    print("SEPARATION ANALYSIS")
    print("=" * 70)
    # Find optimal threshold (balanced accuracy)
    all_scores = np.concatenate([human_scores, ai_scores])
    labels = np.concatenate([np.zeros(len(human_scores)), np.ones(len(ai_scores))])
    best_ba, best_t = 0, 0.5
    for t in np.linspace(0.01, 0.99, 1000):
        preds = (all_scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        tpr = tp / labels.sum()
        tnr = tn / (len(labels) - labels.sum())
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best_ba, best_t = ba, t

    print(f"  Optimal balanced-accuracy threshold: {best_t:.4f} (BA={best_ba:.4f})")

    # At current thresholds
    for name, threshold in [("possible", model.possible), ("high", model.high)]:
        tp = (ai_scores >= threshold).sum()
        fp = (human_scores >= threshold).sum()
        tpr = tp / len(ai_scores)
        fpr = fp / len(human_scores)
        print(f"  At '{name}' ({threshold:.4f}): recall={tpr:.3f}, FPR={fpr:.3f}")

    print()
    print("=" * 70)
    print("PER-QUESTION DETAIL")
    print("=" * 70)
    for i, (question, _, _) in enumerate(CALIBRATION_DATA):
        q_short = question[:50]
        h, a = human_scores[i], ai_scores[i]
        gap = a - h
        flag = ""
        if h >= model.possible:
            flag += " [!human flagged]"
        if a < model.possible:
            flag += " [!AI missed]"
        print(f"  {q_short:50s}  human={h:.3f}  ai={a:.3f}  gap={gap:+.3f}{flag}")

    # Suggested new thresholds
    print()
    print("=" * 70)
    print("SUGGESTED CALIBRATED THRESHOLDS")
    print("=" * 70)
    # "possible" = catches most AI with some false positives (balanced)
    # "high" = very few false positives (conservative)
    p95_human = np.percentile(human_scores, 95)
    p99_human = np.percentile(human_scores, 99)
    print(f"  Human 95th percentile: {p95_human:.4f} (suggested 'possible')")
    print(f"  Human 99th percentile: {p99_human:.4f} (suggested 'high')")
    print(f"  At possible={p95_human:.4f}: AI recall={((ai_scores >= p95_human).sum() / len(ai_scores)):.3f}")
    print(f"  At high={p99_human:.4f}:     AI recall={((ai_scores >= p99_human).sum() / len(ai_scores)):.3f}")


if __name__ == "__main__":
    main()
