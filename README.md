# Plotly
Самодостаточная и интерактивная HTML графика 

Начнём с главного: пользуйтесь важным приложением к материалам репозитория: 
https://soundcloud.com/uchiha-shisui-74300872/koutetsujou-no-kabaneri-ost-grenzlinie-v2

Всем пользоваться и рыдать от милоты, кто не рыдает - марш от сюда! это сказка не для вас.

# Репозиторий повествует о том, что я смог в t-SNE
# но это не главное, главное это Plotly!

Plotly это вещь не хуже, чем тетрадка Юпитер - бибилиотека не только позволяет создавать 
интерактивные графики, но и сохранять их в качестве автономных HTML страниц, которые (!)
работают автономно.

Если пройтись по файлам репозитория: 
Matplotlib и Seaborn>t-SNE>Plotly

То, можно заметить код:

plotly.offline.plot(fig, filename='years_stats.html', show_link=False)

Которые технически звучит так: сохранить как HTML файл filename с графиком на борту.

И - это просто божественно, ибо исполнение этой команды выдаст вам интерактивный график 
в формате HTML страниц, которая откроеться в любом браузере с сохранением функционала.




